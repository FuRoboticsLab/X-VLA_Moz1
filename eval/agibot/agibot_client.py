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
        'arm_init': [
                    -1.098023772239685,
                    0.5286651849746704,
                    0.4588477611541748,
                    -1.2188999652862549,
                    0.5836851000785828,
                    1.3916549682617188,
                    -0.0812821015715599,
                    1.0742743015289307,
                    -0.6114130616188049,
                    -0.27979329228401184,
                    1.2843374013900757,
                    -0.7310851812362671,
                    -1.4948368072509766,
                    0.18800629675388336
                    ],
        'gripper_init': [
                    0.21733333333333293,
                    0.0
                    ],
        'waist_init':  [
                    0.5235992948969566,
                    0.27
                    ],
        'head_init': [1.065264417860243e-05,
                        0.43633763187764485],
    },
    1: {
        'instruction': "Pick objects from the conveyor belt and place them in the box.",
        'arm_init': [
                    -1.1081970930099487,
                    0.5599356293678284,
                    0.41192471981048584,
                    -1.2570282220840454,
                    0.6400485634803772,
                    1.4426263570785522,
                    -0.11550155282020569,
                    1.075513243675232,
                    -0.6112560629844666,
                    -0.28197455406188965,
                    1.283238172531128,
                    -0.7300207614898682,
                    -1.4948891401290894,
                    0.18620894849300385
                    ],
        'gripper_init': [
                    0.21733333333333293,
                    0.0
                    ],
        'waist_init': [
                    0.5235990830818403,
                    0.22
                    ],
        'head_init': [
                    -2.6631610446506076e-06,
                    0.4363349687166002
                    ],
    },
    2: {
        'instruction': "Hang the snacks on the shelf.",
        'arm_init': [
                    -1.3661954402923584,
                    0.9899733662605286,
                    0.7172473669052124,
                    -1.1319290399551392,
                    1.0417301654815674,
                    1.2514617443084717,
                    -0.2595163881778717,
                    1.4043235778808594,
                    -0.936977744102478,
                    -0.9782644510269165,
                    0.41049379110336304,
                    -0.6365411281585693,
                    -1.4750136137008667,
                    -0.26410576701164246
                    ],# 和数采位置对齐
        'gripper_init': [0.0, 0.21733333333333293],
        'waist_init': [
                    0.19198609119071014,
                    0.31
                    ],
        'head_init': [
                    0.0,
                    0.4363323055555555
                    ],
    },
    3: {
        'instruction': "pour the water into the cup.",
        'arm_init': [
                    -1.0356225967407227,
                    0.5868958234786987,
                    0.2758146822452545,
                    -1.2341338396072388,
                    0.7017517685890198,
                    1.4393632411956787,
                    -0.17889739573001862,
                    1.022430419921875,
                    -0.5536535978317261,
                    -0.29788893461227417,
                    1.228410243988037,
                    -0.7772579193115234,
                    -1.4413524866104126,
                    0.15357744693756104
                    ], # 和数采位置对齐
        'gripper_init': [0.21733333333333293, 0.21733333333333293],
        'waist_init': [
                    0.5235961255152174,
                    0.29999801635742185
                    ],
        'head_init': [
                    0.0,
                    0.43630567394510905
                    ],
    },
    4: {
        'instruction': "Open the microwave, put the food in, and close the microwave.",
        'arm_init': [
                    -1.0744252374439445,
                    0.611131855103108,
                    0.2795946672659981,
                    -1.2841009879307974,
                    0.7305285914033108,
                    1.4955309216230965,
                    -0.1876292967659418,
                    1.0744079022701831,
                    -0.611143425100047,
                    -0.2795599866593961,
                    1.28411259741578,
                    -0.7305401614002497,
                    -1.4955540221289296,
                    0.18762929661135214
                    ],# 和数采位置对齐
        'gripper_init': [
                    0.21733333333333293,
                    0.21733333333333293
                    ],
        'waist_init': [
                    0.4363320392394511,
                    0.22
                    ],
        'head_init': [
                    -2.6631610446506076e-06,
                    0.4363296423945109
                    ],
    },
    5: {
        'instruction': "fold the clothes",
        'arm_init': [
                    -1.0743615627288818,
                    0.610994279384613,
                    0.2796013653278351,
                    -1.2838836908340454,
                    0.7304046154022217,
                    1.495360255241394,
                    -0.18758749961853027,
                    1.074326753616333,
                    -0.6110815405845642,
                    -0.2795315384864807,
                    1.2839010953903198,
                    -0.730509340763092,
                    -1.4952731132507324,
                    0.18760496377944946
                    ], # 和数采位置对齐
        'gripper_init': [0.0, 0.0],
        'waist_init': [
                    0.8901176920412174,
                    0.46
                    ],
        'head_init': [
                    0.0,
                    0.4363323055555555
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
        if self.proprio is None: self.proprio = to_flat_array(proprio)    
        if not self.action_plan:
            print(obs['cam_head'].shape)
            query = {
                "proprio": json_numpy.dumps(self.proprio),
                "language_instruction": instruction,
                "image0": json_numpy.dumps(obs['cam_head']),
                "image1": json_numpy.dumps(obs['cam_left_wrist']),
                "image2": json_numpy.dumps(obs['cam_right_wrist']),
                "domain_id": self.domain_id
            }
            response = requests.post(self.url, json=query)
            actions = np.array(response.json()['action'])[self.chunk_size]
            self.action_plan.extend(actions.tolist())
        
        action_predict = np.array(self.action_plan.popleft())
        self.proprio = to_flat_array(proprio) if self.close_loop else action_predict
        action_predict = self.post_process(action_predict)
        return action_predict
    
    def post_process(self, action):
        # proprio: the env proprioception
        # action: the model vanilla output 
        action = to_flat_array(action)
        
        if self.mode == "abs_joint":
            return action
        elif self.mode == "delta_joint":
            left_joint = action[0:7] + self.proprio[0:7]
            right_joint = action[7:14] + self.proprio[7:14]
            left_gripper = action[14]
            right_gripper = action[15]
            return np.concatenate([left_joint, right_joint, left_gripper, right_gripper])
        elif self.mode == "abs_eef":
            right_xyz = action[0:3]
            right_quat = rotate6d_to_quat(action[3:9])
            right_gripper = action[9]
            left_xyz = action[10:13]
            left_quat = rotate6d_to_quat(action[13:19])
            left_gripper = action[19]
            return np.concatenate([right_xyz, right_quat, right_gripper, left_xyz, left_quat, left_gripper])
        elif self.mode == "delta_eef":
            right_xyz = action[0:3] + self.proprio[0:3]
            right_quat = rotate6d_to_quat(action[3:9])
            right_gripper = action[9]
            left_xyz = action[10:13] + self.proprio[10:13]
            left_quat = rotate6d_to_quat(action[13:19])
            left_gripper = action[19]
            return np.concatenate([right_xyz, right_quat, right_gripper, left_xyz, left_quat, left_gripper])
            
            
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
            encoded_str = encode_image(img)
            return img, encoded_str
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
    agent = ClientModel(args.server_ip, args.server_port, args.chunk_size, args.control_mode, args.close_loop, domain_id=domain_id)
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
        print("✅ 系统初始化完成！")
        print("🚀 进入主控制循环...")

        # --- 2. 主控制循环 ---
        count = 0
        while True:
            time.sleep(interval_time)
            print("\n" + "="*50)
            
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

                # print(robot_dds.arm_joint_states()[0].shape)
                joint_pose_state = to_flat_array(robot_dds.arm_joint_states()[0])
                # print('suc')
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
                raw_img, encoded_img = get_and_encode_image(camera, sdk_name)
                encoded_images[server_name] = raw_img # we use raw image
                if raw_img is not None:
                    print(f"✅ 已获取并编码图像: {server_name}")
                else:
                    print(f"⚠️ 无法获取图像: {server_name}")
                if raw_img is not None:
                    log_path = os.path.join(LOG_IMAGE_DIR, f"{server_name}_{timestamp}.png")
                    cv2.imwrite(log_path, raw_img)
                    latest_path = f"./{server_name}_latest.png"
                    cv2.imwrite(latest_path, raw_img)
                    print(f"[Saved] {latest_path}")
            
            # --- 2.4. 解析并执行动作 ---
            if "eef" in args.control_mode:
                action = agent.step(encoded_images, eef_pose_state, current_instruction)
                print(f"[Step {count}] with action: {action}")
                if action.shape[0] != 16:
                    print(f"[!] 动作维度不正确 (应为16)，跳过此动作: {action}")
                    continue

                right_pose_array, left_pose_array = action[0:8], action[8:16]
                gripper_states = [left_pose_array[7].item(), right_pose_array[7].item()]
                right_pose_dict = { "x": right_pose_array[0].item(), 
                                   "y": right_pose_array[1].item(), 
                                   "z": right_pose_array[2].item(), 
                                   "qx": right_pose_array[3].item(), 
                                   "qy": right_pose_array[4].item(), 
                                   "qz": right_pose_array[5].item(), 
                                   "qw": right_pose_array[6].item()}
                left_pose_dict = { "x": left_pose_array[0].item(), 
                                  "y": left_pose_array[1].item(),
                                  "z": left_pose_array[2].item(),
                                  "qx": left_pose_array[3].item(),
                                  "qy": left_pose_array[4].item(),
                                  "qz": left_pose_array[5].item(),
                                  "qw": left_pose_array[6].item() }
            
                robot_controller.set_end_effector_pose_control(
                    lifetime=1.0,
                    control_group=["dual_arm"],
                    right_pose=right_pose_dict,
                    left_pose=left_pose_dict,
                )
                robot_dds.move_gripper(gripper_states)
            elif "joint" in args.control_mode:
                action = agent.step(encoded_images, joint_pose_state, current_instruction)
                print(f"[Step {count}] with action: {action}")
                if action.shape[0] != 16:
                    print(f"[!] 动作维度不正确 (应为16)，跳过此动作: {action}")
                    continue
                right_joints = action[0:7]
                left_joints = action[7:14]
                gripper_states = [action[-2].item(), action[-1].item()]
                # robot_controller.set_joint_position_control( # check this function in A2D SDK
                #     lifetime=1.0,
                #     joint_group={
                #         "left_arm": left_joints.tolist(),
                #         "right_arm": right_joints.tolist()
                #     }
                # )
                robot_dds.move_gripper(gripper_states)
            else:
                print(f"Unsupported control mode: {args.control_mode}")
                break

    except KeyboardInterrupt:
        print("\n[Main] 用户手动中断程序。")
    except Exception as e:
        print(f"\n[Main] ❌ 程序执行时发生严重错误: {e}")
    finally:
        # --- 3. 安全关闭 ---
        if robot_dds:
            print("\n[Main] 重置机器人到安全位置...")
            robot_dds.reset()
            time.sleep(2)
            robot_dds.shutdown()
        if camera:
            camera.close()
        print("[Main] 程序已安全退出。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config for agibot client")
    parser.add_argument("--server_ip", type=str, required=True, help="server ip address")
    parser.add_argument("--server_port", type=int, default=8000, help="server port")
    parser.add_argument("--control_mode", type=str, default="abs_eef", choices=["abs_eef", "delta_eef", "abs_joint", "delta_joint"], help="control mode")
    parser.add_argument("--chunk_size", type=int, default=20, help="number of actions to execute per inference")
    parser.add_argument("--close_loop", action="store_true", help="whether to run in closed-loop mode")
    parser.add_argument('--task_id', type=int, default=0, choices=[0,1,2,3,4,5], help='6 different tasks')
    parser.add_argument('--control_freq', type=int, default=30, help='control frequency (Hz)')

    args = parser.parse_args()
    main(args)