# copy from https://github.com/mees/calvin/blob/main/calvin_models/calvin_agent/evaluation/evaluate_policy.py

import argparse
import collections
from collections import Counter, defaultdict
import logging
import os
from pathlib import Path
import sys
import time
import imageio
import os.path as osp
import json
# This is for using the locally installed repo clone when using slurm
from calvin_agent.models.calvin_base_model import CalvinBaseModel

sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())

from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import (
    collect_plan,
    count_success,
    create_tsne,
    get_default_model_and_env,
    get_env_state_for_initial_condition,
    get_log_dir,
    join_vis_lang,
    print_and_save,
)
from calvin_agent.utils.utils import get_all_checkpoints, get_checkpoints_for_epochs, get_last_checkpoint
import hydra
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from termcolor import colored
import torch
from tqdm.auto import tqdm

from calvin_env.envs.play_table_env import get_env

import json_numpy
import requests
import PIL.Image as Image 

logger = logging.getLogger(__name__)

EP_LEN = 720
NUM_SEQUENCES = 1000
from scipy.spatial.transform import Rotation as R

def euler_xyz_to_rotate6D(q: np.ndarray) -> np.ndarray:
    return R.from_euler('xyz', q, degrees=False).as_matrix()[..., :, :2].reshape(q.shape[:-1] + (6,))

def rotate6D_to_quat(v6: np.ndarray) -> np.ndarray:
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
    quats = R.from_matrix(rot_mats).as_quat()
    return quats


def get_epoch(checkpoint):
    if "=" not in checkpoint.stem: return "0"
    checkpoint.stem.split("=")[1]


def make_env(dataset_path):
    val_folder = Path(dataset_path) / "validation"
    env = get_env(val_folder, show_gui=False)
    return env


class ClientModel(CalvinBaseModel):
    def __init__(self,
                 host,
                 port):

        self.url = f"http://{host}:{port}/act"
        self.action_plan = []
        self.reset()
        
    def reset(self):
        """
        This is called
        """
        # currently, we dont use historical observation, so we dont need this fc
        self.action_plan = []
        self.proprio = None
        return None

    def step(self, obs, goal):
        """
        Args:
            obs: (dict) environment observations
            goal: (str) language goal 
        Returns:
            action: (np.array) predicted action
        """
        if not self.action_plan:
            main_view = obs['rgb_obs']['rgb_static']   # np.ndarray with shape (200, 200, 3)
            wrist_view = obs['rgb_obs']['rgb_gripper']   # np.ndarray with shape (84, 84, 3)
            if self.proprio is None:
                self.proprio = np.concatenate([obs['robot_obs'][:3], 
                                    euler_xyz_to_rotate6D(obs['robot_obs'][3:6]),
                                    obs['robot_obs'][-1:] < 0.5], axis=-1)
                self.proprio = np.concatenate([self.proprio, np.zeros_like(self.proprio)], axis=-1)
            
            query = {"language_instruction": goal,
                     "proprio": json_numpy.dumps(self.proprio.tolist()),
                    "image0": json_numpy.dumps(main_view),
                    "image1": json_numpy.dumps(wrist_view)}
            response = requests.post(self.url, json=query)
            action = response.json()['action']
            self.action_plan.extend(action)
            
        action_predict = np.array(self.action_plan[0])
        
        self.proprio = action_predict
        self.proprio[10:] = 0.
        
        self.action_plan = self.action_plan[1:]
        return (
            action_predict[:3],
            rotate6D_to_quat(action_predict[3:9]),
            1 if action_predict[9] < 0.7 else -1
        )


def evaluate_policy(model, env, epoch, eval_log_dir=None, debug=False, create_plan_tsne=False):
    """
    Run this function to evaluate a model on the CALVIN challenge.

    Args:
        model: Must implement methods of CalvinBaseModel.
        env: (Wrapped) calvin env.
        epoch:
        eval_log_dir: Path where to log evaluation results. If None, logs to /tmp/evaluation/
        debug: If True, show camera view and debug info.
        create_plan_tsne: Collect data for TSNE plots of latent plans (does not work for your custom model)

    Returns:
        Dictionary with results
    """
    # conf_dir = Path(__file__).absolute().parents[0] / "conf"
    conf_dir = Path("/mnt/petrelfs/zhengjinliang/Toolbox/calvin/calvin_env/conf/validation")
    task_cfg = OmegaConf.load(conf_dir / "new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "new_playtable_validation.yaml")

    eval_log_dir = get_log_dir(eval_log_dir)

    eval_sequences = get_sequences(NUM_SEQUENCES)

    results = []
    plans = defaultdict(list)

    if not debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    for initial_state, eval_sequence in eval_sequences:
        result = evaluate_sequence(env, model, task_oracle, initial_state, eval_sequence, val_annotations, plans, debug, eval_log_dir)
        results.append(result)
        if not debug:
            eval_sequences.set_description(
                " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(count_success(results))]) + "|"
            )
        with open(f"{eval_log_dir}/log.txt", 'a+') as f:
            list_r = count_success(results)
            list_r.append(sum(list_r))
            print(" ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(list_r)]) + "|", file=f)

    if create_plan_tsne:
        create_tsne(plans, eval_log_dir, epoch)
    print_and_save(results, eval_sequences, eval_log_dir, epoch)

    return results


def evaluate_sequence(env, model, task_checker, initial_state, eval_sequence, val_annotations, plans, debug, eval_log_dir):
    """
    Evaluates a sequence of language instructions.
    """
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter = 0
    if debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    model.reset()
    for subtask in eval_sequence:
        success, imgs, lang_annotation = rollout(env, model, task_checker, subtask, val_annotations, plans, debug)
        save_path = f'{eval_log_dir}/{lang_annotation}_{success}.mp4'
        save_video(save_path, imgs, fps=30)
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter


def rollout(env, model, task_oracle, subtask, val_annotations, plans, debug):
    """
    Run the actual rollout on one subtask (which is one natural language instruction).
    """
    if debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
    obs = env.get_obs()
    # get lang annotation for subtask
    lang_annotation = val_annotations[subtask][0]
    lang_annotation = lang_annotation.split('\n')[0]
    if '\u2019' in lang_annotation:
        lang_annotation.replace('\u2019', '\'')
    
    start_info = env.get_info()
    imgs = []
    for step in range(EP_LEN):
        action = model.step(obs, lang_annotation)
        obs, _, _, current_info = env.step(action)
        main_view = obs['rgb_obs']['rgb_static']
        H, W, C = main_view.shape
        wrist_view = np.asarray(Image.fromarray(obs['rgb_obs']['rgb_gripper']).resize((H, W)))
        image_obs = np.concatenate([main_view, wrist_view], axis=1)  # np.ndarray with shape (200, 400, 3)
        imgs.append(image_obs)
        
        if debug:
            img = env.render(mode="rgb_array")
            join_vis_lang(img, lang_annotation)
            # time.sleep(0.1)
        if step == 0:
            # for tsne plot, only if available
            collect_plan(model, plans, subtask)

        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            if debug:
                print(colored("success", "green"), end=" ")
            return True, imgs, lang_annotation
    if debug:
        print(colored("fail", "red"), end=" ")
        
    return False, imgs, lang_annotation

def save_video(save_path, images, fps=30):
    imageio.mimsave(save_path, images, fps=fps)

def main():
    seed_everything(0, workers=True)  # type:ignore
    parser = argparse.ArgumentParser(description="Evaluate a trained model on multistep sequences with language goals.")
    parser.add_argument("--output_dir", type=str, help="Where to log the evaluation results.")
    args = parser.parse_args()
    kwargs = vars(args)
    os.makedirs(kwargs['output_dir'], exist_ok=True)
    env = make_env("/mnt/petrelfs/zhengjinliang/Toolbox/calvin/dataset/ABC_D")
    
    while True:
        if not os.path.exists(osp.join(kwargs['output_dir'], 'info.json')): pass
        else: break
    
    time.sleep(5)
    with open(osp.join(kwargs['output_dir'], 'info.json'), 'r') as f:
        infos = json.load(f)
        print(infos)
        host = infos['host']
        port = infos['port']
        job_id = infos['job_id']
    os.remove(osp.join(kwargs['output_dir'], 'info.json'))
    print("-"*88)
    print("save path:", kwargs['output_dir'])
    print("-"*88)
    
    model = ClientModel(host=host, port=port)
    evaluate_policy(model, env, 
                    epoch=None, 
                    eval_log_dir=kwargs['output_dir'], 
                    debug=False)

    import subprocess
    kill_client =  f"""#!/bin/bash
#SBATCH -p mozi_t
#SBATCH -N 1
scancel {job_id}
"""
    job_file = os.path.join(kwargs['output_dir'], f"run_kill.sh")
    subprocess.run(['sbatch', job_file], capture_output=True, text=True)


if __name__ == "__main__":
    main()