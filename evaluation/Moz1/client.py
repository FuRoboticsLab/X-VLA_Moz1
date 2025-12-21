# --------editing by cyn----------

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
import pickle
import requests
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
import torch  # noqa: F401  # (kept in case of future GPU array ops)
# import torchvision.transforms as transforms  # noqa: F401
# from tqdm import tqdm
from utils import euler_to_rotate6d, rotate6d_to_xyz

# moz1 control
from mozrobot import MOZ1Robot,MOZ1RobotConfig

def smooth_action(obs, action_plan, in_freq=30, out_freq=120, trajectory_type="linear", mode="delta"):

    assert len(action_plan) == in_freq and out_freq % in_freq == 0
    num_steps = int(out_freq // in_freq)
    smooth_action_plan = []

    for i in range(len(action_plan)):
        if i == 0:
            if mode == "delta":
                start = np.zeros_like(action_plan[0])
                start[9] = obs["proprio"][9] / 0.12 * 1
                start[19] = obs["proprio"][19] / 0.12 * 1
            else:
                start = obs["proprio"],
            
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
    
    return np.asarray(smooth_action_plan)

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
    def __init__(self, policy, instruction=None, ctrl_freq=120, ctrl_mode="abs", camera=True, infer_interval=0, action_dur=2, action_decay=1/3):
        """
        The Executor for MozRobot. It gets the observation and sent to the XVLA server and then rollout the action.
        Args:
            policy: the instance of `class ClientModel`
            instruction: the text you want the robot to execute
            ctrl_freq: the control freq
            ctrl_mode: [abs, delta], delta will output delta action
            camera: whether camera is connected.
            infer_thres: when the remaining action step is less than thres, post the server to get new action.
            action_dur: the time duration for the generated actions, keep same to the qdur in the training process of XVLA
        """
        if camera:
            config = MOZ1RobotConfig(
                realsense_serials="235422301820, 230322276191, 230422272019",
                structure="wholebody", # choice: wholebody_without_base, wholebody
                robot_control_hz=120,
            )
        else:
            config = MOZ1RobotConfig(
                no_camera=True,
                structure="wholebody", # choice: wholebody_without_base, wholebody
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
        self.getting_action_flag = 0
        self.action_plan_lock = asyncio.Lock()
        self.current_action = None

        self.cmd_pose = []
        self.state_pose = []

        # save the params
        self.ctrl_freq = ctrl_freq
        self.ctrl_mode = ctrl_mode
        self.policy = policy
        self.instruction = instruction
        self.infer_interval = infer_interval
        self.infer_thres = (action_dur - infer_interval) * ctrl_freq
        self.action_dur = action_dur
        self.action_decay = action_decay
        self.action_length = 30

        # monitor command from model
        self.need_monitor = True
        self.max_monitor_length = 2000
        self.action_dim = 6
        self.all_command = np.zeros((self.max_monitor_length, self.max_monitor_length+1+120*action_dur, self.action_dim))
        self.all_state = np.zeros((self.max_monitor_length, self.action_dim))
        self.global_timestep = 0

    async def rollout(self, instruction=None):
        if instruction is not None:
            self.instruction = instruction
            self.global_timestep = 0
        await self.get_policy_action()
        last_left_grip = 0.12
        last_right_grip = 0.12
        try:
            while 1:
                start = time.time()
                if len(self.action_plan) <= self.infer_thres and not self.getting_action_flag:
                    print("----------------Chunk Over.----------------")
                    self.getting_action_flag = 1
                    get_action_task = asyncio.create_task(self.get_policy_action())
                
                try:
                    if self.need_monitor and self.global_timestep < self.max_monitor_length:
                        saved = False
                        obs = self.robot.capture_robot_observation()
                        self.all_state[self.global_timestep] = obs["leftarm_state_cart_pos"]
                    elif self.need_monitor and self.global_timestep > self.max_monitor_length and not saved:
                        with open("record_state.pkl", "wb") as f:
                            pickle.dump(self.all_state, f)
                        saved = True
                    if len(self.action_plan) > self.infer_thres:
                        self.getting_action_flag = 0
                    # else:
                    #     self.getting_action_flag += 1
                    # if self.getting_action_flag > 10:
                    #     await get_action_task
                    await self.action_plan_lock.acquire()
                    action = self.action_plan.popleft()
                    self.action_plan_lock.release()
                    left_arm = action[:10]
                    right_arm = action[10:]
                    print(f"Left Grip: {left_arm[9]}, Right Grip: {right_arm[9]}")
                    # change to 6D Pose + Gripper
                    left_pose = left_arm[:3]
                    left_ori = rotate6d_to_xyz(left_arm[3:9])
                    left_grip = left_arm[9:10] * 0.12
                    # left_grip = np.array([0.12 if left_arm[9] > 0.7 else 0])
                    left_arm = np.concatenate([left_pose, left_ori])
                    
                    right_pose = right_arm[:3]
                    right_ori = rotate6d_to_xyz(right_arm[3:9])
                    right_grip = right_arm[9:10] * 0.12
                    # right_grip = np.array([0.12 if right_arm[9] > 0.7 else 0])
                    right_arm = np.concatenate([right_pose, right_ori])
                    
                    action_dict = {
                        "leftarm_cmd_cart_pos": left_arm,
                        "leftarm_gripper_cmd_pos": left_grip,
                        "rightarm_cmd_cart_pos": right_arm,
                        "rightarm_gripper_cmd_pos": right_grip,
                    }
                    
                    self.robot.send_action(action_dict)
                    self.global_timestep += 1
                    
                        
                    
                    # if np.abs(left_grip-last_left_grip) > 0.06 or np.abs(right_grip-last_right_grip) > 0.06:
                    #     await asyncio.sleep(0.7) # wait for the gripper move
                    # last_left_grip = left_grip
                    # last_right_grip = right_grip
                except Exception as e:
                    print(e)
                    self.action_plan_lock.release()
                finally:
                    await asyncio.sleep(self.time_interval)
                
        finally:
            self.robot.disconnect()
        
    def reset(self):
        self.robot.reset_robot_positions()

    def get_weight(i, total, mode="quick new"):
        if mode == "linear":
            return i / total
        elif mode == "quick new":
            a = 3 * i / total
            return a if a < 1 else 1
        else:
            raise NotImplementedError
    
    async def get_policy_action(self):
        self.getting_action_flag = 1
        await self.action_plan_lock.acquire()
        obs = self.robot.capture_observation()
        start_action_length = len(self.action_plan)
        current_time_step = self.global_timestep
        self.action_plan_lock.release()

        # formulate proprio input
        left_pose = np.asarray(obs["leftarm_state_cart_pos"]) # [6]
        right_pose = np.asarray(obs["rightarm_state_cart_pos"]) # [6]
        left_grip = np.asarray(obs["leftarm_gripper_state_pos"]) / 0.12
        right_grip = np.asarray(obs["rightarm_gripper_state_pos"]) / 0.12
        # left_grip = [1] if np.asarray(obs["leftarm_gripper_state_pos"]) > 0.06 else [0] #  [1]
        # right_grip = [1] if np.asarray(obs["rightarm_gripper_state_pos"]) > 0.06 else [0] #  [1]
        left = np.concatenate([left_pose[:3], euler_to_rotate6d(left_pose[3:]), np.asarray(left_grip)])
        right = np.concatenate([right_pose[:3], euler_to_rotate6d(right_pose[3:]), np.asarray(right_grip)])
        obs["proprio"] = np.concatenate([left, right], -1)

        payload = {}
        payload["proprio"] = json_numpy.dumps(obs["proprio"])
        payload["language_instruction"] = self.instruction
        payload["image0"] = json_numpy.dumps(obs["cam_high"]) # [H, W, 3]
        payload["image1"] = json_numpy.dumps(obs["cam_left_wrist"])
        payload["image2"] = json_numpy.dumps(obs["cam_right_wrist"])
        payload["domain_id"] = 19
        payload["steps"] = 10 # denoising steps

        raw_action_plan = await asyncio.to_thread(self.policy.post, payload)

        # 30 actions to 120Hz*duration actions (if duration=2 seconds, than it is 240)
        if self.ctrl_freq == 120:
            obs = self.robot.capture_observation()
            # formulate proprio input
            left_pose = np.asarray(obs["leftarm_state_cart_pos"]) # [6]
            right_pose = np.asarray(obs["rightarm_state_cart_pos"]) # [6]

            # left_grip = [1] if np.asarray(obs["leftarm_gripper_state_pos"]) > 0.06 else [0] #  [1]
            # right_grip = [1] if np.asarray(obs["rightarm_gripper_state_pos"]) > 0.06 else [0] #  [1]
            left = np.concatenate([left_pose[:3], euler_to_rotate6d(left_pose[3:]), np.asarray(left_grip)])
            right = np.concatenate([right_pose[:3], euler_to_rotate6d(right_pose[3:]), np.asarray(right_grip)])
            obs["proprio"] = np.concatenate([left, right], -1)
            raw_action_plan = smooth_action(obs, raw_action_plan, in_freq=len(raw_action_plan), out_freq=self.ctrl_freq*self.action_dur) 
        
        if self.ctrl_mode == "delta":
            idx = np.array([0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18])
            raw_action_plan[:, idx] += obs["proprio"][idx]
        
        if self.need_monitor and current_time_step < self.max_monitor_length:
            saved = False
            # for now we only monitor left_pose
            leftarm_actions = []
            for action in raw_action_plan:
                left_arm = action[:10]
                left_pose = left_arm[:3]
                left_ori = rotate6d_to_xyz(left_arm[3:9])
                left_arm = np.concatenate([left_pose, left_ori])
                leftarm_actions.append(left_arm)
            leftarm_actions = np.asarray(leftarm_actions)
            self.all_command[current_time_step, current_time_step+1: current_time_step+1+len(leftarm_actions), :] = leftarm_actions
        elif self.need_monitor and current_time_step >= self.max_monitor_length and not saved:
            with open("record_action.pkl", "wb") as f:
                pickle.dump(self.all_command, f)
            saved = True

        await self.action_plan_lock.acquire()
        # infer_thres               |          *    =6
        # initial                   [0 1 2 3 4 5 6 7 8 9]
        # start infer       [0 1 2 3|4 5 6 7 8 9]
        # end infer           [4 5 6|7 8 9]
        #                     [3 2 1|0 1 2 3 4 5 6 7 8 9 10]
        end_action_length = len(self.action_plan)
        bias = end_action_length - start_action_length
        index = np.arange(len(raw_action_plan)) - bias
        for i, action in zip(index, raw_action_plan):
            if i < 0:
                continue
            if i < len(self.action_plan):
                w = i / len(self.action_plan)
                self.action_plan[i] =  (1-w) * self.action_plan[i] + w * action
            else:
                self.action_plan.append(action)
        self.action_plan_lock.release()

        return
    
    def get_observation(self):
        return self.robot.capture_observation()
    
    async def monitor(self):
        await asyncio.sleep(1)
        save_interval = 100
        time_interval = 0.1
        count = 0
        while 1:
            if self.current_action:
                count += 1
            else:
                await asyncio.sleep(time_interval)
                continue
            await self.action_plan_lock.acquire()
            obs = self.robot.capture_observation()
            self.cmd_pose.append(self.current_action["leftarm_cmd_cart_pos"])
            self.state_pose.append(obs["leftarm_state_cart_pos"])
            self.action_plan_lock.release()
            if count % save_interval == 0:
                with open("record.pkl", "wb") as f:
                    pickle.dump({"cmd":self.cmd_pose,"state":self.state_pose}, f)
            await asyncio.sleep(time_interval)
        return 
        # unavailable due to the unacssessibility of matplotlib
        # ===================== 配置参数（可根据需求修改）=====================
        WINDOW_SIZE = 50          # 横轴显示的时间窗口大小（最近50个数据点）
        DIMENSIONS = 6            # 数组维度（固定为6）
        UPDATE_INTERVAL = 0.1     # 每次更新的时间间隔（秒），模拟业务循环耗时
        # Y_AXIS_RANGE = (-5, 5)    # Y轴固定范围（注释掉则自动适配数据）
        Y_AXIS_RANGE = None
        USE_REAL_TIME = False     # 是否使用实际时间戳（False则用循环次数作为时间）

        # ===================== 初始化数据存储 =====================
        # 时间列表（横轴数据）
        time_list = []
        # 6个维度的数据列表（每个维度对应一个列表）
        data_lists = [[] for _ in range(DIMENSIONS)]
        data_lists_2 = [[] for _ in range(DIMENSIONS)]

        # ===================== 创建绘图布局 =====================
        # 创建6个子图，垂直排列，共享X轴，设置画布大小
        fig, axes = plt.subplots(DIMENSIONS, 1, figsize=(12, 2*DIMENSIONS), sharex=True)
        fig.suptitle('Real-time 6-Dimension Variable Monitoring', fontsize=16, y=0.98)

        # 初始化每个子图的折线对象
        lines = []
        lines_2 = []
        for idx, ax in enumerate(axes):
            # 设置子图标题和Y轴标签
            ax.set_title(f'Dimension {idx+1}', fontsize=12, pad=5)
            ax.set_ylabel('Value', fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # 固定Y轴范围（可选）
            if Y_AXIS_RANGE:
                ax.set_ylim(Y_AXIS_RANGE)
            
            # 创建初始空折线
            line, = ax.plot([], [], 'b-', linewidth=1.2, marker='.', markersize=3)
            lines.append(line)
            line, = ax.plot([], [], 'm-', linewidth=1.2, marker='.', markersize=3)
            lines_2.append(line)

        # 设置最后一个子图的X轴标签
        axes[-1].set_xlabel('Time' if USE_REAL_TIME else 'Cycle Index', fontsize=10)

        # 开启matplotlib交互模式（核心：支持实时刷新）
        plt.ion()
        # 调整子图间距，避免重叠
        plt.tight_layout()

        # ===================== 主循环（实时更新）=====================
        cycle_idx = 0  # 循环计数器（作为时间标识，若不用实际时间）
        start_time = time.time()  # 记录程序启动时间（用于实际时间计算）

        while True:
            # -------------------- 1. 生成/获取6维数组新值（替换为你的业务逻辑） --------------------
            # 模拟6维随机数据（正态分布，均值0，方差1），请替换为实际的6维数组
            obs = self.robot.capture_observation()
            new_6d_data = obs["leftarm_cmd_cart_pos"]
            new_6d_data_2 = obs["leftarm_state_cart_pos"]
            
            # -------------------- 2. 更新时间和数据列表 --------------------
            cycle_idx += 1
            # 记录时间（可选：循环次数 或 实际时间戳）
            current_time = time.time() - start_time if USE_REAL_TIME else cycle_idx
            time_list.append(current_time)
            
            # 更新每个维度的数据
            for dim in range(DIMENSIONS):
                data_lists[dim].append(new_6d_data[dim])
                data_lists_2[dim].append(new_6d_data_2[dim])
            
            # -------------------- 3. 裁剪数据（仅保留最近WINDOW_SIZE个点） --------------------
            if len(time_list) > WINDOW_SIZE:
                time_list = time_list[-WINDOW_SIZE:]  # 裁剪时间轴
                for dim in range(DIMENSIONS):
                    data_lists[dim] = data_lists[dim][-WINDOW_SIZE:]  # 裁剪对应维度数据
                    data_lists_2[dim] = data_lists_2[dim][-WINDOW_SIZE:] 
            
            # -------------------- 4. 更新折线图数据 --------------------
            for dim in range(DIMENSIONS):
                # 更新折线的X/Y数据
                lines[dim].set_xdata(time_list)
                lines[dim].set_ydata(data_lists[dim])

                lines_2[dim].set_xdata(time_list)
                lines_2[dim].set_ydata(data_lists_2[dim])
                
                # 若未固定Y轴范围，自动适配Y轴（可选）
                if not Y_AXIS_RANGE:
                    axes[dim].relim()  # 重新计算数据范围
                    axes[dim].autoscale_view(scalex=False, scaley=True)  # 仅自动调整Y轴
            
            # -------------------- 5. 刷新画布 --------------------
            fig.canvas.draw()       # 重绘画布
            fig.canvas.flush_events()  # 刷新事件队列（关键：避免界面卡死）
            
            # -------------------- 6. 模拟业务循环耗时 --------------------
            await asyncio.sleep(UPDATE_INTERVAL)


if __name__ == "__main__":
    # real execution
    policy = ClientModel("10.176.56.103", "8000")

    test = MozExecutor(policy, ctrl_freq=120, ctrl_mode="delta", action_dur=3, infer_interval=2)
    asyncio.run(test.rollout("Fold the red t-shirt."))

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
