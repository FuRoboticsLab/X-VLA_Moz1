# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import os
import random
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path
import subprocess
from accelerate import Accelerator
from datasets import create_dataloader
from xvla import xvla
from safetensors.torch import load_file
from accelerate.utils import DistributedDataParallelKwargs

def submit_openloop_job(model, ckpt_path, output_dir):
    job_script = f"""#!/bin/bash
#SBATCH -p mozi_t
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -o {output_dir}/openloop_log_%j.out
#SBATCH -e {output_dir}/openloop_log_%j.out
source ~/.bashrc
source ~/.latest_env
conda activate UniAct
python -u eval_openloop.py \\
--model {model} \\
--batch_size 16 \\
--checkpoints {ckpt_path} \\
--output_dir {output_dir}
"""
    job_file = os.path.join(output_dir, "run_openloop.sh")
    os.makedirs(output_dir, exist_ok=True)
    with open(job_file, "w") as f: f.write(job_script)
    result = subprocess.run(["sbatch", job_file], capture_output=True, text=True)
    print("Job submitted:", result.stdout.strip())
    
def get_args_parser():
    parser = argparse.ArgumentParser('Training script', add_help=False)
    # Base Settings
    parser.add_argument('--eval_times', default=5, type=int)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--iters', default=1000000, type=int)
    parser.add_argument('--train_metas_path', type=str)
    parser.add_argument('--precision', default='fp16', type=str)
    
    
    parser.add_argument('--model', default='HFP_base', type=str)
    parser.add_argument('--learning_coef', default=1., type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--seed', default=0, type=int)
    
    # Resume & Checkpoint Save & evaluation parameters
    parser.add_argument('--save_interval', default=20000, type=int)
    parser.add_argument('--log_interval', default=10, type=int)
    
    
    parser.add_argument('--output_dir', default='runnings/',
                        help='path where to save, empty for no saving')
    
    parser.add_argument('--resume', default=None, help='model resume from checkpoint')
    parser.add_argument('--pretrained', default=None, help='model load pretraining param')
    
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--port', default=29531, type=int, help='port')
    return parser

def main(args):
    output_dir = Path(args.output_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(mixed_precision = args.precision,
                              log_with="tensorboard", 
                              project_dir=output_dir, kwargs_handlers=[kwargs])
    accelerator.init_trackers("HFP_Training")
    torch.distributed.barrier()
    model = xvla(pretrained = args.pretrained)
    text_processor = model.text_preprocessor
    if args.pretrained is not None:
        accelerator.print('>>>>>> load pretrain from {}'.format(args.pretrained))
        print(model.load_state_dict(load_file(args.pretrained), strict=False))
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000 / 1000
    accelerator.print(f'number of params: {n_parameters} M')
    train_dataloader = iter(create_dataloader(
        batch_size = args.batch_size,
        metas_path = args.train_metas_path,
        num_actions= model.num_actions,
        training=True
    ))
    
    model = model.to(torch.float32)
    encoder_params = list(model.vlm.parameters()) + \
        list(model.transformer.soft_prompt_hub.parameters())
    other_params = filter(lambda p: id(p) not in list(map(id, encoder_params)), model.parameters())
    # 设置优化器参数组
    optim = torch.optim.AdamW([
            {'params': encoder_params, 
                'lr': args.learning_rate * args.learning_coef,
                'weight_decay': 0.},
            
            {'params': other_params,  
                'lr': args.learning_rate,
                'weight_decay': args.weight_decay}
        ],
        betas = (0.9, 0.95)
    )
    model, optim = accelerator.prepare(model, optim)
    if args.resume is not None:
        accelerator.print('>>>>>> resume from {}'.format(args.resume))
        accelerator.load_state(args.resume)
    train_dataloader = iter(train_dataloader)
    model.train()
    accelerator.print(f"Start training for {args.iters} iters")
    for iters in range(args.iters):
        past_time = time.time()
        data = next(train_dataloader)
        language_instruction = text_processor.encode_language(data['language_instruction'])
        del data['language_instruction']
        inputs = {
            **{key: value.cuda(non_blocking=True) for key, value in data.items()},
            **{key: value.cuda(non_blocking=True) for key, value in language_instruction.items()}}
        optim.zero_grad()
        loss_dict = model(**inputs)
        loss = sum(loss_dict.values())
        accelerator.backward(loss)
        optim.step()
        if iters % args.log_interval == 0: 
            accelerator.log({key: value.item() for key, value in loss_dict.items()}, step=iters)
            accelerator.print(f"[Iter {iters}] [Training Loss] {loss.item()} [time_per_iter] {time.time() - past_time}")
            
        if iters % args.save_interval == 0:
            model.eval()
            accelerator.print("========start saving models=========")
            accelerator.save_state(os.path.join(output_dir, f"last_ckpt"), safe_serialization=True)
            accelerator.save_model(model, os.path.join(output_dir, f"ckpt-{iters}"), safe_serialization=True)
            accelerator.print(f"[Iter {iters}] Start Openloop evaluation")
            if accelerator.is_main_process:
                submit_openloop_job(args.model, 
                                   os.path.join(output_dir, f"last_{iters}"), 
                                   os.path.join(output_dir, f"ckpt-{iters}"))
            
            accelerator.wait_for_everyone()
            model.train()
        accelerator.wait_for_everyone()
    accelerator.save_model(model, os.path.join(output_dir, f"ckpt-final"), safe_serialization=True)
    
def slurm_env_init(args):
    args.rank = int(os.environ['SLURM_PROCID'])
    args.gpu = args.rank % torch.cuda.device_count()
    args.world_size = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    addr = subprocess.getoutput(
        f'scontrol show hostname {node_list} | head -n1')
    os.environ['MASTER_PORT'] = str(getattr(args, 'port', '29529'))
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(args.world_size)
    os.environ['LOCAL_RANK'] = str(args.rank % num_gpus)
    os.environ['RANK'] = str(args.rank)
    torch.cuda.set_device(args.gpu)
    
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    
    # fix the seed for reproducibility
    seed = args.seed + torch.distributed.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    return args

if __name__ == '__main__':
    parser = argparse.ArgumentParser('training script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(slurm_env_init(args))