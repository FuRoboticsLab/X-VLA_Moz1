
import argparse
import model
from safetensors.torch import load_file
from timm import create_model
import simpler_env
import os
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from simpler_env.utils.visualization import write_video
import numpy as np
from scipy.spatial.transform import Rotation as R
from sapien.core import Pose
import torch
from PIL import Image
import math

def quat_to_rotate6D(q: np.ndarray) -> np.ndarray:
	return R.from_quat(q).as_matrix()[..., :, :2].reshape(q.shape[:-1] + (6,))

def rotate6D_to_xyz(v6: np.ndarray) -> np.ndarray:
    v6 = np.asarray(v6)
    if v6.shape[-1] != 6:
        raise ValueError("Last dimension must be 6 (got %s)" % (v6.shape[-1],))
    a1 = v6[..., 0:5:2]
    a2 = v6[..., 1:6:2]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 - proj
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    rot_mats = np.stack((b1, b2, b3), axis=-1)      # shape (..., 3, 3)
    return R.from_matrix(rot_mats).as_euler('xyz')


def evaluate_policy_widowX(model, text_processor, image_processor, eval_log_dir, proc_id, max_steps = 40):
    
    for task in [
                'widowx_spoon_on_towel', 
                 'widowx_carrot_on_plate', 
                 'widowx_stack_cube', 
                 'widowx_put_eggplant_in_basket']:
        print(f"Eval Task: {task} for proc {proc_id}")
        images = []
        env = simpler_env.make(task)
        obs, _ = env.reset(options={"obj_init_options": {"episode_id": proc_id}})
        instruction = env.get_language_instruction()
        
        ee_pose_wrt_base = Pose(p=obs['agent']['base_pose'][:3], q=obs['agent']['base_pose'][3:]).inv() * Pose(p=obs['extra']['tcp_pose'][:3], q=obs['extra']['tcp_pose'][3:])
        proprio =  torch.from_numpy(np.concatenate([ee_pose_wrt_base.p, np.array([1,0,0,1,0,0,0])], axis=-1)).to(dtype=torch.float32)
        proprio = torch.cat([proprio, torch.zeros_like(proprio)], dim = -1)
        for _ in range(max_steps):
            image = get_image_from_maniskill2_obs_dict(env, obs)
            language_inputs  = text_processor.encode_language([instruction])
            image_inputs = image_processor([Image.fromarray(image)])
            print("current_proprio:", proprio)
            inputs = {
                **{key: value.cuda(non_blocking=True) for key, value in language_inputs.items()},
                **{key: value.cuda(non_blocking=True) for key, value in image_inputs.items()},
                'proprio':  proprio.unsqueeze(0).cuda(non_blocking=True),
                'hetero_info': torch.tensor(0).unsqueeze(0).cuda(non_blocking=True),
                'steps': 10
            }
            with torch.no_grad(): action = model.pred_action(**inputs)[0]
            proprio = action[-1, :10]
            proprio = torch.cat([proprio, torch.zeros_like(proprio)], dim = -1)
            for a in action.cpu().numpy():
                obs, reward, done, _, _ = env.step(np.concatenate(
                    [a[:3], 
                     rotate6D_to_xyz(a[3:9]) + np.array([0, math.pi /2, 0]), 
                     np.array([1]) if a[9] < 0.7 else np.array([-1])
                     ]))
                image = get_image_from_maniskill2_obs_dict(env, obs)
                images.append(image.copy())
                if done: break
            if done: break
        write_video(f"{eval_log_dir}/{instruction}_{proc_id}_{reward}.mp4", images, fps=10)
        with open(os.path.join(eval_log_dir, f"widowx_results.txt"), "a+") as f:
            f.write(f"{task}, {proc_id}, {done}\n")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser('Training script', add_help=False)
    # Base Settings
    parser.add_argument('--eval_times', default=24, type=int)
    parser.add_argument('--model', default='HFP_base', type=str)
    parser.add_argument('--checkpoints', type=str, help='model checkpoints')
    parser.add_argument('--output_dir', type=str)
    args = parser.parse_args()
    
    path = f"{args.checkpoints}/model.safetensors"
    print(f"load ckpt from {path}")
    ckpt = load_file(path)
    
    model, text_processor, image_preprocessor = create_model(args.model)
    print(model.load_state_dict(ckpt, strict=False))
    model = model.to(torch.float32).cuda()
    for i in range(args.eval_times):
        evaluate_policy_widowX(model,
                            text_processor,
                            image_preprocessor,
                            eval_log_dir=args.output_dir,
                            proc_id=i)