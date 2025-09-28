
from mmengine import fileio
import io
import os
import os.path as osp
import torch
import torchvision.transforms as transforms
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv
import robosuite.utils.transform_utils as T
from tqdm import tqdm
import json
import imageio
import collections
import requests
import argparse
import json_numpy
import time
import subprocess

class LiberoAbsActionProcessor:
    def Rotate6D_to_AxisAngle(self, r6d):
        """
        r6d: np.ndarray, shape (N, 6)
        return: np.ndarray, shape (N, 3), axis-angle vectors
        """
        flag = 0
        if len(r6d.shape) == 1:
            r6d = r6d[None, ...]
            flag = 1
            
        a1 = r6d[:, 0:3]
        a2 = r6d[:, 3:6]
        
        # b1
        b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-6)
        
        # b2
        dot_prod = np.sum(b1 * a2, axis=-1, keepdims=True)
        b2_orth = a2 - dot_prod * b1
        b2 = b2_orth / (np.linalg.norm(b2_orth, axis=-1, keepdims=True) + 1e-6)
        
        # b3
        b3 = np.cross(b1, b2, axis=-1)
        
        R = np.stack([b1, b2, b3], axis=-1)  # shape: (N, 3, 3)
        
        axis_angle_list = []
        for i in range(R.shape[0]):
            quat = T.mat2quat(R[i])
            axis_angle = T.quat2axisangle(quat)
            axis_angle_list.append(axis_angle)
        
        axis_angle_array = np.stack(axis_angle_list, axis=0)  # shape: (N, 3)
        
        if flag == 1:   
            axis_angle_array = axis_angle_array[0]
                    
        return axis_angle_array
            
    def Mat_to_Rotate6D(self, abs_action):
        if len(abs_action.shape) == 2:
            return np.concatenate([abs_action[:3, 0], abs_action[:3, 1]], axis=-1)
        elif len(abs_action.shape) == 3:
            return np.concatenate([abs_action[:, :3, 0], abs_action[:, :3, 1]], axis=-1)
        else:
            raise NotImplementedError
        
    def AxisAngle_to_Rotate6D(self, abs_action):
        if len(abs_action.shape) == 1:
            return self.Mat_to_Rotate6D(T.quat2mat(T.axisangle2quat(abs_action)))
        else:
            raise NotImplementedError
        
    def action_6d_to_axisangle(self, abs_action):
        if len(abs_action.shape) == 1:
            final_ori = self.Rotate6D_to_AxisAngle(abs_action[3:9])
            result_action = np.concatenate([abs_action[0:3], final_ori, abs_action[-1:]])
            return result_action
        elif len(abs_action.shape) == 2:
            final_ori = self.Rotate6D_to_AxisAngle(abs_action[:, 3:9])
            result_action = np.concatenate([abs_action[:, 0:3], final_ori, abs_action[:, -1:]])
            return result_action
        else:
            raise NotImplementedError

EPS = 1e-5
LIBERO_DATASETS = {'libero_goal': ["libero_goal"],
                   "libero_object": ["libero_object"],
                   "libero_spatial": ["libero_spatial"],
                   "libero_10": ["libero_10"],
                   "libero_90": ["libero_90"],
                   "libero30": ["libero_goal", "libero_object", "libero_spatial"],
                   "libero130": ["libero_goal", "libero_object", "libero_spatial", "libero_10", "libero_90"]}

LIBERO_DATASETS_HORIZON = {
    "libero_goal": 800,
    "libero_object": 800,
    "libero_spatial": 800,
    "libero_10": 900,
    "libero_90": 800,
    "libero30": 800,
    "libero130": 800,
}

benchmark_dict = benchmark.get_benchmark_dict()

class ClientModel():
    def __init__(self,
                 host,
                 port):

        self.url = f"http://{host}:{port}/act"
        self.processor = LiberoAbsActionProcessor()
        self.reset()
        
        
    def reset(self):
        """
        This is called
        """
        # currently, we dont use historical observation, so we dont need this fc
        self.proprio = None
        self.action_plan = collections.deque()
        return None

    def step(self, obs, goal):
        if not self.action_plan:
            main_view = np.flip(np.flip(obs["agentview_image"], 0), 1)  # np.ndarray with shape (256, 256, 3)
            wrist_view = obs["robot0_eye_in_hand_image"]   # np.ndarray with shape (256, 256, 3)
                        
            # prioprio
            # if self.proprio is None:
            self.proprio = np.concatenate([obs['robo_pos'], obs['robo_ori'], np.array([0])], axis=-1)
            self.proprio = np.concatenate([self.proprio, np.zeros_like(self.proprio)], axis=-1)
            query = {
                "proprio": json_numpy.dumps(self.proprio),
                "language_instruction": goal,
                "image0": json_numpy.dumps(main_view),
                "image1": json_numpy.dumps(wrist_view)
            }
            response = requests.post(self.url, json=query)
            action = np.array(response.json()['action'])
            # self.proprio = action[-1]
            target_eef = action[:, :3]
            target_axis = self.processor.Rotate6D_to_AxisAngle(action[:, 3:9])
            target_act = action[:, 9:10]
            final_action = np.concatenate([target_eef, target_axis, target_act], axis=-1)
            self.gripper = final_action[-1, -1:]
            self.action_plan.extend(final_action.tolist())
    
        action_predict = np.array(self.action_plan.popleft())
        action_predict[-1] = 1 if action_predict[-1] > 0.5 else -1
        return action_predict

class LIBEROEval():
    def __init__(self, task_suite_name: str, eval_horizon: int=600, act_type:str = 'abs',
                num_episodes: int=10, eval_freq: int=10, init_seed: int=42):
        
        self.task_suite_name = task_suite_name
        self.task_list = LIBERO_DATASETS[self.task_suite_name]
        self.task_suite_list = [benchmark_dict[task]() for task in self.task_list]
        self.eval_horizon = eval_horizon
        self.num_episodes = num_episodes
        self.eval_freq = eval_freq
        self.init_seed = init_seed
        self.act_type = act_type
        self.processor = LiberoAbsActionProcessor()

    def _make_dir(self, save_path):
        task_suite_name = self.task_suite_name
        path = os.path.join(save_path, task_suite_name)
        if not os.path.exists(path):
            os.makedirs(path)
        self.base_dir = path
    
    def _init_env(self, task_suite, task_id=0, ep=0):
        # get task information and env args
        task = task_suite.get_task(task_id)
        task_name = task.name
        task_description = task.language
        task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        print(f"[info] retrieving task {task_id} from suite {self.task_suite_name}, the " + \
                f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")

        # step over the environment
        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": 256,
            "camera_widths": 256
        }
        
        # init thesubprocess vector environment
        env = OffScreenRenderEnv(**env_args)
        
        # environment reset 
        env.seed(self.init_seed + ep + 100)
        obs = env.reset()
        init_states = task_suite.get_task_init_states(task_id) # for benchmarking purpose, we fix the a set of initial states
        init_state_id = ep % init_states.shape[0]
        obs = env.set_init_state(init_states[init_state_id])
        
        for i in range(10):
            action = np.array([0., 0., 0., 0., 0., 0., -1.0])
            obs, reward, done, info = env.step(action)
        
        if self.act_type == 'abs':
            for robot in env.env.robots:
                robot.controller.use_delta=False
        elif self.act_type == 'rel':
            pass
        else:
            raise NotImplementedError

        return env, task_description, obs 
    
    def _log_results(self, metrics: dict):
        print(metrics)
        save_name = os.path.join(self.base_dir, 'results.json')
        with open(save_name, 'a+') as f:
            line = json.dumps(metrics)
            f.write(line+'\n')

    def _save_video(self, save_path: str, images: list, done: list, fps=30): 
        imageio.mimsave(save_path, images, fps=fps)

    def _rollout(self, task_suite, policy, task_id, ep):
        env, lang, obs  = self._init_env(task_suite, task_id, ep)

        images = []
        for t in tqdm(range(self.eval_horizon), desc=f'{lang}'):
            robo_ori = self.processor.Mat_to_Rotate6D(env.env.robots[0].controller.ee_ori_mat)
            robo_pos = env.env.robots[0].controller.ee_pos
            obs['robo_ori'] = robo_ori
            obs['robo_pos'] = robo_pos
    
            action = policy.step(obs, lang)
            
            images.append(np.flip(np.flip(obs['agentview_image'], 0), 1))
            obs, reward, done, info = env.step(action)
            if done:
                break

        save_path = f'{self.base_dir}/{lang}_{ep}.mp4'
        self._save_video(save_path, images, done, fps=30)
        
        success = float(done)
        metrics = {f'sim/{self.task_suite_name}/{lang}': success}
        self._log_results(metrics)
        
        env.close()
        return success
    
    def eval_episodes(self, policy, save_path: str):
        """
        rollout several episodes and log the mean episode return
        """
        self._make_dir(save_path)
        
        rews = []
        for task_suite in self.task_suite_list:
            for task_id in tqdm(range(len(task_suite.tasks)), desc="Evaluating..."):
                for ep in range(self.num_episodes):
                    policy.reset()                    
                    rew = self._rollout(task_suite, policy, task_id, ep)
                    rews.append(rew)
        eval_rewards = sum(rews) / len(rews)
        metrics = {f'sim_summary/{self.task_suite_name}/all': eval_rewards}
        self._log_results(metrics)
        return eval_rewards


def eval_libero(agent, save_path, num_episodes=10, init_seed=42, act_type='abs',
                task_suites=["libero_goal", "libero_spatial", "libero_10"]):

    result_dict = {}
    for suite_name in task_suites:
        horizon = LIBERO_DATASETS_HORIZON[suite_name]
        evaluator = LIBEROEval(task_suite_name=suite_name, eval_horizon=horizon, act_type=act_type,
                           num_episodes=num_episodes, init_seed=init_seed)
        eval_rewards = evaluator.eval_episodes(agent, save_path=save_path)
        result_dict[suite_name] = eval_rewards
    with open(f"{save_path}/results.json", "a+") as f:
        json.dump(result_dict, f, indent=4)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='single-process evaluation on Libero bench')
    parser.add_argument('--task_suites', default=['libero_10', 'libero_spatial', 'libero_goal', 'libero_object'], nargs='+', help='base save path')
    parser.add_argument('--eval_time', default=1, type=int, help='evaluate episodes')
    parser.add_argument('--init_seed', default=42, type=int, help='base save path')
    parser.add_argument('--output_dir', type=str, help='base save path')
    args = parser.parse_args()
    kwargs = vars(args)
    os.makedirs(kwargs['output_dir'], exist_ok=True)
    
    while True:
        if not os.path.exists(osp.join(kwargs['output_dir'], 'info.json')):
            pass
        else:
            break
   
    time.sleep(5)
    with open(osp.join(kwargs['output_dir'], 'info.json'), 'r') as f:
        infos = json.load(f)
        print(infos)
        host = infos['host']
        port = infos['port']
        job_id = infos['job_id']
    os.remove(osp.join(kwargs['output_dir'], 'info.json'))
    print("-"*88)
    print("init seed:", kwargs['init_seed'])
    print("save path:", kwargs['output_dir'])
    print("task suites:", kwargs['task_suites'])
    print("evaluate episodes:", kwargs['eval_time'])
    print("-"*88)

    agent = ClientModel(host=host, port=port)
    import subprocess
    kill_client =  f"""#!/bin/bash
#SBATCH -p mozi_t
#SBATCH -N 1
scancel {job_id}
"""
    job_file = os.path.join(kwargs['output_dir'], f"run_kill.sh")
    with open(job_file, "w") as f: f.write(kill_client)
    try:
        eval_libero(agent=agent, save_path=kwargs['output_dir'], init_seed=kwargs['init_seed'],
                        num_episodes=kwargs['eval_time'], task_suites=kwargs['task_suites'], act_type='abs')
    except: subprocess.run(['sbatch', job_file], capture_output=True, text=True)
    subprocess.run(['sbatch', job_file], capture_output=True, text=True)