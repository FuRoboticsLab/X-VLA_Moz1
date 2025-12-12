# --------editing by cyn----------
from __future__ import annotations

import argparse
import asyncio
import collections
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import json_numpy
import numpy as np
import requests
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
import torch  # noqa: F401  # (kept in case of future GPU array ops)
# import torchvision.transforms as transforms  # noqa: F401
# from tqdm import tqdm
from utils import euler_to_rotate6d, rotate6d_to_xyz

# moz1 control
from mozrobot import MOZ1Robot,MOZ1RobotConfig

def smooth_action(action_plan, in_freq=30, out_freq=120, trajectory_type="linear"):

    assert len(action_plan) == in_freq and out_freq % in_freq == 0
    num_steps = out_freq // in_freq
    smooth_action_plan = []

    for i in range(len(action_plan)):
        if i == 0:
            start = np.zeros_like(action_plan[0])
        else:
            start = action_plan[i - 1]
        end = action_plan[i]
        for j in range(num_steps):
            if trajectory_type == "cosine":
                # Smooth cosine interpolation
                alpha = 0.5 * (1 - np.cos(np.pi * (j+1) / (num_steps)))
            elif trajectory_type == "cubic":
                # Cubic easing
                t = (j+1) / (num_steps)
                alpha = t * t * (3 - 2 * t)
            else:  # linear
                alpha =(j+1) / (num_steps)
            smooth_action_plan.append(start + alpha * (end - start))
    
    return smooth_action_plan

class ClientModel:
    """Thin HTTP client that queries a remote policy server and returns actions."""

    def __init__(self, host: str, port: int):
        self.url = f"http://{host}:{port}/act"
        self.reset()


    def reset(self) -> None:
        """重置客户端状态"""
        self.proprio: Optional[np.ndarray] = None  # last absolute [pos(3)+ori6d(6)+grip(1)]
        self.action_plan: Deque[List[float]] = collections.deque()

    def post(self, payload: Dict) -> np.ndarray:
        try:
            resp = requests.post(self.url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            raise RuntimeError(f"Policy server request failed: {e}") from e

        action = np.array(data["action"])  # shape (T, 10) expected: [pos3, rot6d, grip1]
        if action.ndim != 2 or action.shape[1] < 10:
            raise RuntimeError(f"Unexpected action shape from server: {action.shape}")
        return action
    
class MozExecutor:
    def __init__(self, policy, instruction=None, ctrl_freq=120, ctrl_mode="abs", camera=True, infer_thres=0):
        """
        The Executor for MozRobot. It gets the observation and sent to the XVLA server and then rollout the action.
        Args:
            policy: the instance of `class ClientModel`
            instruction: the text you want the robot to execute
            ctrl_freq: the control freq
            ctrl_mode: [abs, delta], delta will output delta action
            camera: whether camera is connected.
            infer_thres: when the remaining action step is less than thres, post the server to get new action.
        """
        if camera:
            config = MOZ1RobotConfig(
                realsense_serials="235422301820, 230322276191, 230422272019",
                structure="wholebody", # choice: dualarm, wholebody_without_base, wholebody
                robot_control_hz=120,
            )
        else:
            config = MOZ1RobotConfig(
                no_camera=True,
                structure="wholebody", # choice: dualarm, wholebody_without_base, wholebody
                robot_control_hz=120,
            )

        self.robot = MOZ1Robot(config)
        self.robot.connect()
        if self.robot.is_robot_connected:
            print("机器人连接成功")
        else:
            print("机器人连接不成功，请调试")
            raise RuntimeError("robot connect failed")
        self.robot.enable_external_following_mode()
        self.action_plan = collections.deque()
        self.time_interval = 1.0 / ctrl_freq
        self.ctrl_mode = ctrl_mode
        self.policy = policy
        self.instruction = instruction
        self.getting_action_flag = 0
        self.infer_thres = infer_thres

    async def rollout(self, instruction=None):
        if instruction is not None:
            self.instruction = instruction
        self.get_policy_action()
        try:
            while 1:
                start = time.time()
                if len(self.action_plan) <= self.infer_thres and not self.getting_action_flag:
                    asyncio.create_task(self.get_policy_action())
                try:
                    action = self.action_plan.popleft()
                    end = time.time()
                    await asyncio.sleep(self.time_interval-(end-start))
                    self.robot.send_action(action)
                except:
                    pass
                
        finally:
            self.robot.disconnect()
        
    def reset(self):
        self.robot.reset_robot_positions()
    
    async def get_policy_action(self):
        self.getting_action_flag = 1
        # test asyncio

        obs = self.robot.capture_observation()

        # formulate proprio input
        left_pose = np.asarray(obs["leftarm_state_cart_pos"]) # [6]
        right_pose = np.asarray(obs["rightarm_state_cart_pos"]) # [6]
        left_grip = np.asarray(obs["leftarm_gripper_state_pos"]) #  [1]
        right_grip = np.asarray(obs["rightarm_gripper_state_pos"]) #  [1]
        left = np.concatenate([left_pose[:3], euler_to_rotate6d(left_pose[3:]), left_grip])
        right = np.concatenate([right_pose[:3], euler_to_rotate6d(right_pose[3:]), right_grip])
        obs["proprio"] = np.concatenate([left, right], -1)

        payload = {}
        payload["proprio"] = json_numpy.dumps(obs["proprio"])
        payload["language_instruction"] = self.instruction
        payload["image0"] = json_numpy.dumps(obs["cam_high"]) # [H, W, 3]
        payload["image1"] = json_numpy.dumps(obs["cam_left_wrist"])
        payload["image2"] = json_numpy.dumps(obs["cam_right_wrist"])
        payload["domain_id"] = 19
        payload["steps"] = 10 # denoising steps

        raw_action_plan = self.policy.post(payload)

        raw_action_plan = smooth_action(raw_action_plan)
        
        if self.ctrl_mode == "delta":
            idx = np.array([0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18])
            raw_action_plan[:, idx] += obs["proprio"][idx]

        for action in raw_action_plan:
            left_arm = action[:10]
            right_arm = action[10:]
            # change to 6D Pose + Gripper
            left_pose = left_arm[:3]
            left_ori = rotate6d_to_xyz(left_arm[3:9])
            left_grip = np.array([0.12 if left_arm[9] > 0.06 else 0])
            left_arm = np.concatenate([left_pose, left_ori])
            
            right_pose = right_arm[:3]
            right_ori = rotate6d_to_xyz(right_arm[3:9])
            right_grip = np.array([0.12 if right_arm[9] > 0.06 else 0])
            right_arm = np.concatenate([right_pose, right_ori])
            
            action_dict = {
                "leftarm_cmd_cart_pos": left_arm,
                "leftarm_gripper_cmd_pos": left_grip,
                "rightarm_cmd_cart_pos": right_arm,
                "rightarm_gripper_cmd_pos": right_grip,
            }
            self.action_plan.append(action_dict)
        self.getting_action_flag = 0
        return
    
    def get_observation(self):
        return self.robot.capture_observation()
    
    # def observation_to_command(self, obs):
    #     cmd = {}
    #     for k, v in obs.items():
    #         if "state" in k:
    #             k.replace("state", "cmd")
    #             cmd[k] = v
    #     return cmd
    # def __del__(self):
    #     self.robot.disconnect()
    #     super().__del__()

if __name__ == "__main__":
    # real execution
    policy = ClientModel("10.176.56.103", "8000")
    test = MozExecutor(policy, ctrl_freq=1)
    asyncio.run(test.rollout("Fold the blue t-shirt."))

    # test offline movement
    from copy import deepcopy
    test = MozExecutor(None, camera=True)
    obs = test.get_observation()
    cart_pos = obs["leftarm_state_cart_pos"]
    # obs = test.observation_to_command(obs)
    for i in range(10):
        cmd = {}
        for k, v in obs.items():
            v = deepcopy(v)
            if "gripper" not in k and "cart" in k:
                
                v[0] = v[0] + 0.1 * (i+1)                
                k = k.replace("state", "cmd")
                cmd[k] = v
        test.action_plan.append(cmd)
    
    asyncio.run(test.rollout())
