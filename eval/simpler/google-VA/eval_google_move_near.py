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
import numpy as np
from itertools import product
from sapien.core import Pose
from transforms3d.euler import euler2quat
import itertools

def parse_range_tuple(t):
    # 如果输入是单个数值，直接返回 [t]
    if isinstance(t, (int, float)):
        return [t]
    # 否则认为是 [start, end, num]
    return np.linspace(t[0], t[1], int(t[2])).tolist()

def generate_robot_init_quats(quat_center, rpy_range):
    r_range = parse_range_tuple(rpy_range[:3])
    p_range = parse_range_tuple(rpy_range[3:6])
    y_range = parse_range_tuple(rpy_range[6:])
    return [
        (Pose(q=euler2quat(r, p, y)) * Pose(q=quat_center)).q
        for r, p, y in product(r_range, p_range, y_range)
    ]

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


agg_dict = {
'google_robot_move_near_agg_0' : {
        "task_name": "google_robot_move_near",
        "robot_name": "google_robot_static",
        "env_name": "MoveNearGoogleInScene-v0",           
        "scene_name": "google_pick_coke_can_1_v4",  
        "max_episode_steps": 80,     
        "episode_nums": 60,
        "robot_init_x": 0.35,
        "robot_init_y": 0.21,
        "obj_variation_mode": "episode",
        "robot_init_rot_quat_center": [0, 0, 0, 1],
        "robot_init_rot_rpy_range": [0, 0, 1, 0, 0, 1, -0.09, -0.09, 1],
        "additional_env_build_kwargs": {}
},
'google_robot_move_near_agg_1' : {
        "task_name": "google_robot_move_near",
        "robot_name": "google_robot_static",
        "env_name": "MoveNearGoogleInScene-v0",           
        "scene_name": "google_pick_coke_can_1_v4",  
        "max_episode_steps": 80,     
        "episode_nums": 60,     
        "robot_init_x": 0.35,
        "robot_init_y": 0.21,
        "obj_variation_mode": "episode",
        "robot_init_rot_quat_center": [0, 0, 0, 1],
        "robot_init_rot_rpy_range": [0, 0, 1, 0, 0, 1, -0.09, -0.09, 1],
        "additional_env_build_kwargs": {"no_distractor": True}
},
'google_robot_move_near_agg_2' : {
        "task_name": "google_robot_move_near",
        "robot_name": "google_robot_static",
        "env_name": "MoveNearGoogleInScene-v0",           
        "scene_name": "google_pick_coke_can_1_v4_alt_background",     
        "max_episode_steps": 80,     
        "episode_nums": 60,  
        "robot_init_x": 0.35,
        "robot_init_y": 0.21,
        "obj_variation_mode": "episode",
        "robot_init_rot_quat_center": [0, 0, 0, 1],
        "robot_init_rot_rpy_range": [0, 0, 1, 0, 0, 1, -0.09, -0.09, 1],
        "additional_env_build_kwargs": {}
},
'google_robot_move_near_agg_3' : {
        "task_name": "google_robot_move_near",
        "robot_name": "google_robot_static",
        "env_name": "MoveNearGoogleInScene-v0",           
        "scene_name": "google_pick_coke_can_1_v4_alt_background_2",      
        "max_episode_steps": 80,     
        "episode_nums": 60, 
        "robot_init_x": 0.35,
        "robot_init_y": 0.21,
        "obj_variation_mode": "episode",
        "robot_init_rot_quat_center": [0, 0, 0, 1],
        "robot_init_rot_rpy_range": [0, 0, 1, 0, 0, 1, -0.09, -0.09, 1],
        "additional_env_build_kwargs": {}
},
'google_robot_move_near_agg_4' : {
        "task_name": "google_robot_move_near",
        "robot_name": "google_robot_static",
        "env_name": "MoveNearGoogleInScene-v0",           
        "scene_name": "google_pick_coke_can_1_v4",       
        "max_episode_steps": 80,     
        "episode_nums": 60,
        "robot_init_x": 0.35,
        "robot_init_y": 0.21,
        "obj_variation_mode": "episode",
        "robot_init_rot_quat_center": [0, 0, 0, 1],
        "robot_init_rot_rpy_range": [0, 0, 1, 0, 0, 1, -0.09, -0.09, 1],
        "additional_env_build_kwargs": {"slightly_darker_lighting": True}
},
'google_robot_move_near_agg_5' : {
        "task_name": "google_robot_move_near",
        "robot_name": "google_robot_static",
        "env_name": "MoveNearGoogleInScene-v0",           
        "scene_name": "google_pick_coke_can_1_v4",   
        "max_episode_steps": 80,     
        "episode_nums": 60,    
        "robot_init_x": 0.35,
        "robot_init_y": 0.21,
        "obj_variation_mode": "episode",
        "robot_init_rot_quat_center": [0, 0, 0, 1],
        "robot_init_rot_rpy_range": [0, 0, 1, 0, 0, 1, -0.09, -0.09, 1],
        "additional_env_build_kwargs": {"slightly_brighter_lighting": True}
},
'google_robot_move_near_agg_6' : {
        "task_name": "google_robot_move_near",
        "robot_name": "google_robot_static",
        "env_name": "MoveNearGoogleInScene-v0",           
        "scene_name": "Baked_sc1_staging_objaverse_cabinet1_h870",  
        "max_episode_steps": 80,     
        "episode_nums": 60,     
        "robot_init_x": 0.35,
        "robot_init_y": 0.21,
        "obj_variation_mode": "episode",
        "robot_init_rot_quat_center": [0, 0, 0, 1],
        "robot_init_rot_rpy_range": [0, 0, 1, 0, 0, 1, -0.09, -0.09, 1],
        "additional_env_build_kwargs": {}
},
'google_robot_move_near_agg_7' : {
        "task_name": "google_robot_move_near",
        "robot_name": "google_robot_static",
        "env_name": "MoveNearGoogleInScene-v0",           
        "scene_name": "Baked_sc1_staging_objaverse_cabinet2_h870",  
        "max_episode_steps": 80,     
        "episode_nums": 60,     
        "robot_init_x": 0.35,
        "robot_init_y": 0.21,
        "obj_variation_mode": "episode",
        "robot_init_rot_quat_center": [0, 0, 0, 1],
        "robot_init_rot_rpy_range": [0, 0, 1, 0, 0, 1, -0.09, -0.09, 1],
        "additional_env_build_kwargs": {}
},
'google_robot_move_near_agg_8' : {
        "task_name": "google_robot_move_near",
        "robot_name": "google_robot_static",
        "env_name": "MoveNearAltGoogleCameraInScene-v0",           
        "scene_name": "google_pick_coke_can_1_v4",       
        "max_episode_steps": 80,     
        "episode_nums": 60,
        "robot_init_x": 0.35,
        "robot_init_y": 0.21,
        "obj_variation_mode": "episode",
        "robot_init_rot_quat_center": [0, 0, 0, 1],
        "robot_init_rot_rpy_range": [0, 0, 1, 0, 0, 1, -0.09, -0.09, 1],
        "additional_env_build_kwargs": {}
},
'google_robot_move_near_agg_9' : {
        "task_name": "google_robot_move_near",
        "robot_name": "google_robot_static",
        "env_name": "MoveNearAltGoogleCamera2InScene-v0",           
        "scene_name": "google_pick_coke_can_1_v4",  
        "max_episode_steps": 80,     
        "episode_nums": 60,     
        "robot_init_x": 0.35,
        "robot_init_y": 0.21,
        "obj_variation_mode": "episode",
        "robot_init_rot_quat_center": [0, 0, 0, 1],
        "robot_init_rot_rpy_range": [0, 0, 1, 0, 0, 1, -0.09, -0.09, 1],
        "additional_env_build_kwargs": {}
}
}
 
def evaluate_policy_Google(model, text_processor, image_processor, eval_log_dir, chunk_length = 10):
#     for task in [
#             'google_robot_move_near_agg_0',
#             'google_robot_close_drawer',
#             'google_robot_open_drawer',
#         'google_robot_pick_coke_can',
#    'google_robot_move_near',
#    'google_robot_place_in_closed_drawer',
#   ]:
        
    for task in agg_dict.keys():
        agg_dict_current = agg_dict[task]
        max_steps = agg_dict_current["max_episode_steps"] // 3
        # max_steps = 10
        agg_dict_current["control_freq"] = 3
        agg_dict_current["sim_freq"] = 513
        robot_init_quats = generate_robot_init_quats(
            agg_dict_current["robot_init_rot_quat_center"], 
            agg_dict_current["robot_init_rot_rpy_range"]
        )
        if "rgb_overlay_path" not in agg_dict_current:
            if "rgb_overlay_cameras" not in agg_dict_current:
                if "google_robot_static" in agg_dict_current["robot_name"]:
                    agg_dict_current["rgb_overlay_cameras"] = ["overhead_camera"]
                    
        print(f"length of robot_init_quats: {len(robot_init_quats)}")
        for robot_init_x in parse_range_tuple(agg_dict_current["robot_init_x"]):
            agg_dict_current["robot_init_x"] = robot_init_x
            for robot_init_y in parse_range_tuple(agg_dict_current["robot_init_y"]):
                agg_dict_current["robot_init_y"] = robot_init_y
                for robot_init_quat in robot_init_quats:
                    agg_dict_current["robot_init_rot_quat"] = robot_init_quat
                    
                    make_kwargs = dict(
                        robot=agg_dict_current["robot_name"],
                        sim_freq=agg_dict_current["sim_freq"],
                        control_freq=agg_dict_current["control_freq"],
                        control_mode="arm_pd_ee_base_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner",
                        scene_name=agg_dict_current["scene_name"],
                        camera_cfgs={"add_segmentation": True},
                        rgb_overlay_path=agg_dict_current.get("rgb_overlay_path", None)
                    )
                
                    images = []
                    env = simpler_env.make(agg_dict_current["env_name"], **make_kwargs, **agg_dict_current["additional_env_build_kwargs"])
                    
                    options = {
                        "robot_init_options": {
                            "init_xy": np.array([agg_dict_current["robot_init_x"], agg_dict_current["robot_init_y"]]),
                            "init_rot_quat": robot_init_quat,
                        }
                    }
                    
                    reset_combinations = []
                    
                    if agg_dict_current["obj_variation_mode"] == "episode":
                        for ep_id in range(agg_dict_current["episode_nums"]):
                            reset_combinations.append(
                                {
                                    "episode_id": ep_id
                                })
                    elif agg_dict_current["obj_variation_mode"] == "xy":
                        x_list = parse_range_tuple(agg_dict_current["obj_init_x_range"])
                        y_list = parse_range_tuple(agg_dict_current["obj_init_y_range"])
                        xy_combinations = list(itertools.product(x_list, y_list))
                        for x, y in xy_combinations:
                            reset_combinations.append(
                                {
                                    "init_xy": np.array([x, y])
                                })
                            
                    # main loop
                    ep_count = 0
                    for obj_reset_option in reset_combinations:
                        options["obj_init_options"] = obj_reset_option
                        obs, _ = env.reset(options=options)

                        print(f"Eval Task: {task} for ep count {ep_count}")
                        ee_pose_wrt_base = Pose(p=obs['agent']['base_pose'][:3], q=obs['agent']['base_pose'][3:]).inv() * Pose(p=obs['extra']['tcp_pose'][:3], q=obs['extra']['tcp_pose'][3:])
                        current_xyz = torch.tensor(ee_pose_wrt_base.p).cuda()
                        proprio = torch.zeros(20).to(dtype=torch.float32)
                        images = []
                        
                        for _ in range(max_steps):
                            image = get_image_from_maniskill2_obs_dict(env, obs)
                            instruction = env.get_language_instruction()
                            language_inputs  = text_processor.encode_language([instruction])
                            image_inputs = image_processor([Image.fromarray(image)])
                        #     print("current_proprio:", proprio)
                        #     print("language_inputs:", instruction)
                            inputs = {
                                **{key: value.cuda(non_blocking=True) for key, value in language_inputs.items()},
                                **{key: value.cuda(non_blocking=True) for key, value in image_inputs.items()},
                                'proprio':  proprio.unsqueeze(0).cuda(non_blocking=True),
                                'hetero_info': torch.tensor(1).unsqueeze(0).cuda(non_blocking=True),
                                'steps': 10
                            }
                            with torch.no_grad():
                                action = model.pred_action(**inputs)[0, ::2][:6]
                                # print(action[0, :, :10], current_xyz)
                                action[:, :3] += current_xyz.view(1, 3)
                                # current_xyz = action[-1, :3] open-loop?
                        #     proprio = torch.cat([action[-1, :10], torch.zeros_like(action[-1, :10])], dim = -1)
                            for a in action.cpu().numpy():
                                # print("current_action:", a)
                                obs, reward, done, _, _ = env.step(np.concatenate(
                                    [a[:3],
                                        rotate6D_to_xyz(a[3:9]),
                                        np.array([1]) if a[9] > 0.2 else np.array([-1])
                                        ]))
                                image = get_image_from_maniskill2_obs_dict(env, obs)
                                images.append(image.copy())
                                ee_pose_wrt_base = Pose(p=obs['agent']['base_pose'][:3], q=obs['agent']['base_pose'][3:]).inv() * Pose(p=obs['extra']['tcp_pose'][:3], q=obs['extra']['tcp_pose'][3:])
                                current_xyz = torch.tensor(a[:3]).cuda()
                                if done: break
                            if done: break
                        write_video(f"{eval_log_dir}/{task}_{ep_count}_{reward}.mp4", images, fps=10)
                        with open(os.path.join(eval_log_dir, f"google_results.txt"), "a+") as f:
                            f.write(f"{task}, {ep_count}, {done}\n")
                        ep_count += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Training script', add_help=False)
    # Base Settings
    parser.add_argument('--eval_times', default=20, type=int)
    parser.add_argument('--model', default='HFP_large_32', type=str)
    parser.add_argument('--checkpoints', type=str, default='/home/dodo/.zh1hao_space/proj1/HeteroDiffusionPolicy/AbsEEFFlowV8/exp/rt1', help='model checkpoints')
    parser.add_argument('--output_dir', type=str, default="/home/dodo/.zh1hao_space/proj1/HeteroDiffusionPolicy/AbsEEFFlowV8/logs-open")
    args = parser.parse_args()
    
    path = f"{args.checkpoints}/model.safetensors"
    print(f"load ckpt from {path}")
    ckpt = load_file(path)
    
    model, text_processor, image_preprocessor = create_model(args.model)
    print(model.load_state_dict(ckpt, strict=False))
    model = model.to(torch.float32).cuda()
    # for i in range(args.eval_times):
    evaluate_policy_Google(model, 
                        text_processor, 
                        image_preprocessor,
                        eval_log_dir=args.output_dir,)