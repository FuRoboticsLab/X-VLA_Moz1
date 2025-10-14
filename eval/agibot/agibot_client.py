import base64
import io
import time
import os
import numpy as np
import requests
import cv2
import datetime
from typing import Tuple, Union, Optional
import argparse
import collections
import json_numpy
from datasets.common import quat_to_rotate6d, rotate6d_to_quat
# 从新机器人SDK导入
from a2d_sdk.robot import RobotDds, RobotController, CosineCamera
# --- 全局配置 ---
LOG_IMAGE_DIR = "./Log"  # 保存周期性图像日志的文件夹

CAMERA_MAPPING = {
    "cam_head": "head",
    "cam_left_wrist": "hand_left",
    "cam_right_wrist": "hand_right",
}

TASK_INFOS = {
    0: {
        'instruction': "Pick up the object and place it in the bag.",
        'gripper_threshold': 0.5,
        'chunk_size': 10,
        'arm_init': [
                    -1.11224556,  0.53719825,  0.45914441, -1.23825192,  0.5959, 1.41219366, -0.08660435,
                    1.07460594, -0.61097687, -0.2804215, 1.28363943, -0.72993356, -1.4951334, 0.18722105,
                    ],
        'gripper_init': [
                    0.0,
                    0.0
                    ],
        'waist_init':  [
                    0.52359929,
                    27,
                    ],
        'head_init': [
                    0.0,
                    0.436332306,],
    },
    1: {
        'instruction': "Pick objects from the conveyor belt and place them in the box.",
        'gripper_threshold': 0.5,
        'chunk_size': 10,
        'arm_init': [
                    -1.085, 0.5951, 0.3214, -1.279, 0.7025, 1.479, -0.1656,
                    1.075, -0.6117, -0.2797, 1.282, -0.7310, -1.495, 0.1868,
                    ],
        'gripper_init': [
                    0.0,
                    0.0
                    ],
        'waist_init': [
                    0.5236,
                    26.0,
                    ],
        'head_init': [
                    0.0,
                    0.4363,
                    ],
    },
    2: {
        'instruction': "Hang the snacks on the shelf.",
        'gripper_threshold': 0.5,
        'chunk_size': 15,
        'arm_init': [
                    -1.686, 0.9457, 1.330, -0.8735, 0.1478, 1.252, 0.03354,
                    1.815, -0.6138, -1.393, 0.8388, -0.1517, -1.466, -0.06992,
                    ],# 和数采位置对齐
        'gripper_init': [0.0, 0.0],
        'waist_init': [
                    0.1920, 
                    31
                    ],
        'head_init': [
                    0.0,
                    0.4363,
                    ],
    },
    3: {
        'instruction': "pour the water into the cup.",
        'gripper_threshold': 0.4,
        'chunk_size': 40,
        'arm_init': [
                    -1.03729784, 0.58743685, 0.27705365, -1.23694324, 0.70110613, 1.44067192, -0.17989205,
                    1.07420456, -0.611099, -0.27960137, 1.28388369, -0.73043954, -1.49543011, 0.1876224,
                    ], # 和数采位置对齐
        'gripper_init': [0.0, 0.0],
        'waist_init': [
                    0.52359771, 
                    29.999931
                    ],
        'head_init': [
                    0.0,
                    0.43633231,
                    ],
    },
    4: {
        'instruction': "Open the microwave, put the food in, and close the microwave.",
        'gripper_threshold': 0.5,
        'chunk_size': 30,
        'arm_init': [
                    -1.0742743, 0.61099428, 0.279549, -1.28383136, 0.73043954, 1.49532545, -0.1876224,
                    1.07420456, -0.61097687, -0.2795839, 1.28395355, -0.73038721, -1.49534285, 0.18760496,
                    ],# 和数采位置对齐
        'gripper_init': [
                    0.0,
                    0.0
                    ],
        'waist_init': [
                    0.43633204,
                    24.0
                    ],
        'head_init': [
                    0.0,
                    0.43633231,
                    ],
    },
    5: {
        'instruction': "fold the clothes",
        'gripper_threshold': 0.5,
        'chunk_size': 30,
        'arm_init': [
                    -1.0742219686508179, 0.6111513376235962, 0.27946174144744873, -1.2839535474777222, 0.7303872108459473, 1.495360255241394, -0.18760496377944946,
                    1.0742743015289307, -0.6110466122627258, -0.2794792056083679, 1.2838836908340454, -0.7303697466850281, -1.4952380657196045, 0.18762239813804626,
                    ], # 和数采位置对齐
        'gripper_init': [0.0, 0.0],
        'waist_init': [
                    0.8901176920412174, 
                    45.98676300048828
                    ],
        'head_init': [
                    0.0,
                    0.4363323055555555,
                    ],
    }
}

def to_flat_array(x, dtype=np.float32):
    return np.asarray(x, dtype=dtype).ravel()

class ClientModel():
    def __init__(self,
                 host,
                 port,
                 chunk_size=20,
                 control_mode='abs_eef',
                 close_loop=False,
                 domain_id=20):

        self.url = f"http://{host}:{port}/act"
        self.reset()
        self.chunk_size = chunk_size
        self.control_mode = control_mode
        self.close_loop = close_loop
        self.domain_id = domain_id
        
    def reset(self):
        self.proprio = None
        self.action_plan = collections.deque()
        return None

    def step(self, obs, proprio, instruction):
           
        if not self.action_plan:
            print(self.domain_id, instruction)
            # print(obs['cam_head'].shape)
            if self.proprio is None: self.proprio = to_flat_array(proprio)
            if self.close_loop: self.proprio = to_flat_array(proprio) 
            query = {
                "proprio": json_numpy.dumps(self.proprio),
                "language_instruction": instruction,
                "image0": json_numpy.dumps(obs['cam_head']),
                "image1": json_numpy.dumps(obs['cam_left_wrist']),
                "image2": json_numpy.dumps(obs['cam_right_wrist']),
                "domain_id": self.domain_id,
                "steps": 5
            }
            start_time = time.time()
            print("send query")
            response = requests.post(self.url, json=query)
            print("=======================")
            print(f"query time: ", time.time() - start_time)
            print("=======================")
            actions = np.array(response.json()['action'])[:self.chunk_size]
            actions = self.post_process(actions)
            self.action_plan.extend(actions.tolist())
        
        action_predict = np.array(self.action_plan.popleft())
        if not self.close_loop: 
            if 'ee' in self.control_mode:
                self.proprio = np.concatenate([
                    action_predict[:3],
                    quat_to_rotate6d(action_predict[3:7]),
                    action_predict[7:8],
                    action_predict[8:11],
                    quat_to_rotate6d(action_predict[11:15]),
                    action_predict[15:16]], axis=-1)
            else: self.proprio = action_predict
        # action_predict = self.post_process(action_predict)
        return action_predict
    
    def post_process(self, action):
        # proprio: the env proprioception
        # action: the model vanilla output 
        
        if self.control_mode == "abs_joint":
            return action
        elif self.control_mode == "delta_joint":
            left_joint = action[:, 0:7] + self.proprio[None, 0:7]
            right_joint = action[:, 7:14] + self.proprio[None, 7:14]
            left_gripper = action[:, 14:15]
            right_gripper = action[:, 15:16]
            return np.concatenate([left_joint, right_joint, left_gripper, right_gripper], axis=-1)
        elif self.control_mode == "abs_eef":
            right_xyz = action[:, 0:3]
            right_quat = rotate6d_to_quat(action[:, 3:9])
            right_gripper = action[:, 9:10]
            left_xyz = action[:, 10:13]
            left_quat = rotate6d_to_quat(action[:, 13:19])
            left_gripper = action[:, 19:20]
            return np.concatenate([left_xyz, left_quat, left_gripper, right_xyz, right_quat, right_gripper], axis=-1)
        elif self.control_mode == "delta_eef":
            left_xyz = action[:, 0:3] + self.proprio[None, 0:3]
            left_quat = rotate6d_to_quat(action[:, 3:9], scalar_first=False)
            left_gripper = action[:, 9:10]
            
            right_xyz = action[:, 10:13] + self.proprio[None, 10:13]
            right_quat = rotate6d_to_quat(action[:, 13:19], scalar_first=False)
            right_gripper = action[:, 19:20]

            return np.concatenate([left_xyz, left_quat, left_gripper, right_xyz, right_quat, right_gripper], axis=-1)
            
            
def encode_image(img: np.ndarray) -> str:
    """将 OpenCV 图像编码为 base64 PNG 字符串。"""
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

def clear_log_directory():
    """清空 LOG_IMAGE_DIR 文件夹中的所有文件"""
    if os.path.exists(LOG_IMAGE_DIR):
        for filename in os.listdir(LOG_IMAGE_DIR):
            file_path = os.path.join(LOG_IMAGE_DIR, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"已删除文件: {file_path}")
            except Exception as e:
                print(f"警告：无法删除文件 {file_path}: {e}")
    else:
        os.makedirs(LOG_IMAGE_DIR)
        print(f"📂 已创建文件夹: {LOG_IMAGE_DIR}")

def get_and_encode_image(camera: CosineCamera, cam_sdk_name: str) -> Tuple[np.ndarray, Optional[str]]:
    """
    从指定的摄像头获取图像，返回原始图像和编码后的字符串。
    """
    try:
        img, _ = camera.get_latest_image(cam_sdk_name)
        if img is not None and img.size > 0:
            # encoded_str = encode_image(img)
            return img, None
        else:
            print(f"警告：无法获取 {cam_sdk_name} 的图像，或者图像为空。")
            return None, None
    except Exception as e:
        print(f"警告：获取 {cam_sdk_name} 图像时发生异常: {e}")
        return None, None

def main(args):
    """主程序：连接机器人，并进入主控制循环"""
    if not os.path.exists(LOG_IMAGE_DIR):
        os.makedirs(LOG_IMAGE_DIR)
        print(f"📂 Directory has been made: {LOG_IMAGE_DIR}")
        # 清空 Log 文件夹
    print("🧹 Sweep logs...")
    clear_log_directory()

    robot_dds = None
    robot_controller = None
    camera = None
    domain_id = int(args.task_id) + 20  # AGIBOT domain_id is 15
    
    chunk_size = TASK_INFOS[args.task_id]['chunk_size']
        
    agent = ClientModel(args.server_ip, args.server_port, chunk_size, args.control_mode, args.close_loop, domain_id=domain_id)
    interval_time = 1.0 / args.control_freq

    try:
        # --- 1. 初始化机器人和相机 ---
        print("🤖 Init robot...")
        robot_dds = RobotDds()
        robot_controller = RobotController()
        
        # 从 CAMERA_MAPPING 获取所有需要使用的摄像头SDK名称
        camera_sdk_names = list(CAMERA_MAPPING.values())
        print(f"📷 Init cameras: {camera_sdk_names}")
        camera = CosineCamera(camera_sdk_names)
        
        robot_dds.reset(arm_positions=TASK_INFOS[args.task_id]['arm_init'],
                        gripper_positions=TASK_INFOS[args.task_id]['gripper_init'],
                        hand_positions=robot_dds.hand_initial_joint_position,
                        waist_positions=TASK_INFOS[args.task_id]['waist_init'],
                        head_positions=TASK_INFOS[args.task_id]['head_init'])
        current_instruction = TASK_INFOS[args.task_id]['instruction']
        print(f"📝 Current task_id: {args.task_id}, instruction: {current_instruction}")
        print("✅ 系统初始化完成！INPUT ENTER")
        input()
        print("🚀 进入主控制循环...")

        # --- 2. 主控制循环 ---
        count = 0
        while True:
            time.sleep(interval_time)
            # print("\n" + "="*50)
            
            # --- 2.1. 获取状态和图像 ---
            try:

                motion_status = robot_controller.get_motion_status()
                left_cartesian = motion_status["frames"]["arm_left_link7"]
                right_cartesian = motion_status["frames"]["arm_right_link7"]
                #a2d_sdk.gripper_states() 返回 ([左爪状态, 右爪状态], [时间戳]) 待确认 ? 已确认
                gripper_states_raw, _ = robot_dds.gripper_states() 
                left_gripper_state = gripper_states_raw[0]
                right_gripper_state = gripper_states_raw[1]
                left_6d = quat_to_rotate6d(np.array([left_cartesian["orientation"]["quaternion"]["x"], 
                                            left_cartesian["orientation"]["quaternion"]["y"], 
                                            left_cartesian["orientation"]["quaternion"]["z"], 
                                            left_cartesian["orientation"]["quaternion"]["w"]]))
                
                right_6d = quat_to_rotate6d(np.array([right_cartesian["orientation"]["quaternion"]["x"], 
                                             right_cartesian["orientation"]["quaternion"]["y"], 
                                             right_cartesian["orientation"]["quaternion"]["z"], 
                                             right_cartesian["orientation"]["quaternion"]["w"]]))
                # 16维: left xyz + left 6d + left gripper + right xyz + right 6d + right gripper
                eef_pose_state = np.concatenate([
                    np.array([left_cartesian["position"]["x"], 
                              left_cartesian["position"]["y"], 
                              left_cartesian["position"]["z"]]),
                    left_6d, 
                    np.array([left_gripper_state]),
                    np.array([right_cartesian["position"]["x"], 
                            right_cartesian["position"]["y"], 
                            right_cartesian["position"]["z"]]),
                    right_6d, 
                    np.array([right_gripper_state]),
                ])

                joint_pose_state = to_flat_array(robot_dds.arm_joint_states()[0])
                joint_pose_state = np.concatenate([joint_pose_state, 
                                                   to_flat_array(left_gripper_state), 
                                                   to_flat_array(right_gripper_state)]) # 7+7+1+1=16维

            except (KeyError, IndexError) as e:
                 print(f"❌ 获取机器人状态失败: {e}。跳过本轮循环。")
                 continue
            except Exception as e:
                 print(f"❌ 获取机器人状态时发生未知错误: {e}。跳过本轮循环。")
                 continue
            
            # 获取、编码并保存图像
            encoded_images = {}
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            for server_name, sdk_name in CAMERA_MAPPING.items():
                raw_img, _ = get_and_encode_image(camera, sdk_name)
                
                raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB) if raw_img is not None else None
                raw_img = cv2.resize(raw_img, (236, 236))
                encoded_images[server_name] = cv2.imencode('.jpg', raw_img)[1] # we use raw image
                # if raw_img is not None:
                #     print(f"✅ 已获取并编码图像: {server_name}")
                # else:
                #     print(f"⚠️ 无法获取图像: {server_name}")
                # if raw_img is not None:
                #     log_path = os.path.join(LOG_IMAGE_DIR, f"{server_name}_{timestamp}.png")
                #     cv2.imwrite(log_path, raw_img)
                #     latest_path = f"./{server_name}_latest.png"
                #     cv2.imwrite(latest_path, raw_img)
                #     print(f"[Saved] {latest_path}")
            
            # --- 2.4. 解析并执行动作 ---
            if "eef" in args.control_mode:
                action = agent.step(encoded_images, eef_pose_state, current_instruction)
                left_pose_array, right_pose_array = action[0:8], action[8:16]
                gripper_states = [left_pose_array[7].item(), right_pose_array[7].item()]
                left_pose_dict = { "x": left_pose_array[0].item(), 
                                  "y": left_pose_array[1].item(),
                                  "z": left_pose_array[2].item(),
                                  "qx": left_pose_array[3].item(),
                                  "qy": left_pose_array[4].item(),
                                  "qz": left_pose_array[5].item(),
                                  "qw": left_pose_array[6].item() }
                
                right_pose_dict = { "x": right_pose_array[0].item(), 
                                   "y": right_pose_array[1].item(), 
                                   "z": right_pose_array[2].item(), 
                                   "qx": right_pose_array[3].item(), 
                                   "qy": right_pose_array[4].item(), 
                                   "qz": right_pose_array[5].item(), 
                                   "qw": right_pose_array[6].item()}
                
                robot_controller.set_end_effector_pose_control(
                    lifetime=1.0,
                    control_group=["dual_arm"],
                    right_pose=right_pose_dict,
                    left_pose=left_pose_dict,
                )
            elif "joint" in args.control_mode:
                action = agent.step(encoded_images, joint_pose_state, current_instruction)
                if action.shape[0] != 16:
                    print(f"[!] 动作维度不正确 (应为16)，跳过此动作: {action}")
                    continue
                left_joints = action[0:7]
                right_joints = action[7:14]
                gripper_states = [action[-2].item(), action[-1].item()]
                robot_controller.set_joint_position_control( # check this function in A2D SDK
                    lifetime=1.0,
                    joint_group={
                        "left_arm": left_joints.tolist(),
                        "right_arm": right_joints.tolist()
                    }
                ) 
            else:
                print(f"Unsupported control mode: {args.control_mode}")
                break
            
            gripper_threshold = 0.5
            if 'gripper_threshold' in TASK_INFOS[args.task_id].keys(): 
                gripper_threshold = TASK_INFOS[args.task_id]['gripper_threshold']
            gripper_states = [35 if x < gripper_threshold else 120 for x in gripper_states] # to angle
            robot_dds.move_gripper(gripper_states)

    except KeyboardInterrupt:
        print("\n[Main] 用户手动中断程序。")
    except Exception as e:
        print(f"\n[Main] ❌ 程序执行时发生严重错误: {e}")
    finally:
        # --- 3. 安全关闭 ---
        if robot_dds:
            print("\n[Main] 重置机器人到安全位置...")
            robot_dds.reset()
            # time.sleep(2)
            # robot_dds.shutdown()
        if camera: camera.close()
        print("[Main] 程序已安全退出。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config for agibot client")
    parser.add_argument("--server_ip", type=str, required=True, help="server ip address")
    parser.add_argument("--server_port", type=int, default=8000, help="server port")
    parser.add_argument("--control_mode", type=str, default="abs_eef", choices=["abs_eef", "delta_eef", "abs_joint", "delta_joint"], help="control mode")
    parser.add_argument("--close_loop", action="store_true", help="whether to run in closed-loop mode")
    parser.add_argument('--task_id', type=int, default=0, choices=[0,1,2,3,4,5], help='6 different tasks')
    parser.add_argument('--control_freq', type=int, default=50, help='control frequency (Hz)')

    args = parser.parse_args()
    main(args)