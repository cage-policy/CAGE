import argparse
import os
import time
from multiprocessing import Barrier

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

from agent import CAGEAgent
from hardware.camera import RealSenseRGBDCamera
from hardware.gripper import DahuanAG95Gripper
from hardware.robot import FlexivRobot
from utils.constants import *
from utils.transforms import rot6d_to_rotation_matrix, rotation_matrix_to_rot6d


def rot_diff(mat1, mat2):
    diff = np.matmul(mat1, mat2.T)
    diff = np.diag(diff).sum()
    diff = min(max((diff-1)/2, -1), 1)
    diff = np.arccos(diff)
    return diff

def discrete_rotation(start_rot, end_rot, step=np.pi/16):
    start_6d = rotation_matrix_to_rot6d(start_rot)
    end_6d = rotation_matrix_to_rot6d(end_rot)
    diff = rot_diff(start_rot, end_rot)
    n_step = int(diff//step + 1)
    rot_list = []
    for i in range(n_step):
        rot6d_i = start_6d * (n_step-1-i) + end_6d * (i+1)
        rot6d_i /= n_step
        rot_list.append(rot6d_to_rotation_matrix(rot6d_i))
    return rot_list

def main(args):
    ctrl_freq = args.ctrl_freq

    base_conf = OmegaConf.load(os.path.join('configs', args.config+'.yaml'))
    # merge with model conf in case of using training config
    model_conf_path = os.path.join('configs', 'model', base_conf.model.name+'.yaml')
    if os.path.exists(model_conf_path):
        model_conf = OmegaConf.load(model_conf_path)
        base_conf = OmegaConf.merge(base_conf, model_conf)
    # merge conf with args
    conf = OmegaConf.merge(base_conf, OmegaConf.create(vars(args)))

    # synchronize all devices to the same time
    step_sync = Barrier(
        1+1+1+conf.dataset.meta_data.fixed_views+conf.dataset.meta_data.in_hand_views
    )

    print('[Main] Init Robotic Arm')
    ready_xyz = [0.4, 0.0, 0.17]
    ready_rot = np.array([
        [-1.,  0.,  0.],
        [ 0.,  1.,  0.],
        [ 0.,  0., -1.]
    ])

    robot = FlexivRobot(robot_ip_address='192.168.2.100', pc_ip_address='192.168.2.35', step_sync=step_sync)
    xyz, rot = robot.get_tcp_pose(real_time=True)
    xyz[2] = 0.17
    robot.send_tcp_pose(xyz, rot)
    time.sleep(1)
    robot.send_tcp_pose(ready_xyz, ready_rot)
    # default is [10,10,10, 5,5,5]
    robot.set_max_contact_wrench(np.array([10,10,10, 4,4,4]))

    print('[Main] Init Gripper')
    gripper = DahuanAG95Gripper('/dev/ttyUSB0', step_sync=step_sync)
    gripper.set_force(0.3)

    print('[Main] Init cameras')
    global_camera_ids = ['750612070851']
    wrist_camera_ids = ['043322070878']

    global_cams = []
    global_cam_obs = []
    for i in range(conf.dataset.meta_data.fixed_views):
        global_cams.append(
            RealSenseRGBDCamera(global_camera_ids[i], step_sync=step_sync)
        )
        global_cam_obs.append([])
    wrist_cams = []
    wrist_cam_obs = []
    for i in range(conf.dataset.meta_data.in_hand_views):
        wrist_cams.append(
            RealSenseRGBDCamera(wrist_camera_ids[i], step_sync=step_sync)
        )
        wrist_cam_obs.append([])

    print('[Main] Init agent')
    agent = CAGEAgent(conf, blocking=False, temporal_ensemble=args.t_ensemble, k=[0.05, 0.05, None])    # no smoothing on width
    
    # init obs_dict
    obs_dict = {}
    for i in range(len(global_cams)):
        init_img = Image.fromarray(global_cams[i].get_observation())
        global_cam_obs[i] = [init_img.copy() for _ in range(conf.dataset.obs_horizon)]
        obs_dict[f'global_cam_{i}'] = global_cam_obs[i]
    for i in range(len(wrist_cams)):
        init_img = Image.fromarray(wrist_cams[i].get_observation())
        wrist_cam_obs[i] = [init_img.copy() for _ in range(conf.dataset.obs_horizon)]
        obs_dict[f'wrist_cam_{i}'] = wrist_cam_obs[i]

    proprio = []
    xyz, rot = robot.get_tcp_pose()
    gripper_width = gripper.get_info()[0]
    for _ in range(conf.dataset.obs_horizon if conf.model.use_proprio else 1):
        proprio.append(np.concatenate([
            xyz, rot.flatten(), [gripper_width],
        ]).astype(np.float32))
    obs_dict['proprio'] = proprio
    
    try:
        step_time = 1.0 / ctrl_freq

        prev_width = None
        prev_rot = None
        for step in range(args.max_timestep):
            start_time = time.time()
            # synchronize all devices
            step_sync.wait()
            # yield to let all devices to update their status first
            time.sleep(0.01)

            # 1. update observations
            for i in range(len(global_cams)):
                img = Image.fromarray(global_cams[i].get_observation())
                global_cam_obs[i].append(img)
                global_cam_obs[i].pop(0)
            for i in range(len(wrist_cams)):
                img = Image.fromarray(wrist_cams[i].get_observation())
                wrist_cam_obs[i].append(img)
                wrist_cam_obs[i].pop(0)

            xyz, rot = robot.get_tcp_pose()
            gripper_width = gripper.get_info()[0]
            proprio.append(np.concatenate([
                xyz, rot.flatten(), [gripper_width],
            ]).astype(np.float32))
            proprio.pop(0)

            if step % args.pred_interval == 0:
                agent.update_actions(obs_dict, step)

            # 2. get the action at current step
            xyz, rot, width = agent.get_action(step)
            
            # 3. post-process actions
            xyz = xyz.clip(TRANS_MIN, TRANS_MAX)
            xyz = xyz * SCALE + OFFSET
            xyz[2] = max(xyz[2], 0.002)

            if width <= 0.005:
                width = 0
            width = width * 1.2

            print('\033[91m', end='')   # red
            print(f'{xyz.round(4)}', end=' ')
            # print('\033[93m', end='')   # yellow
            # print(f'({action_tcp[3]:.2f}, {action_tcp[4]:.2f}, {action_tcp[5]:.2f}, {action_tcp[6]:.2f})', end=' ')
            print('\033[92m', end='')   # green
            print(f'{width:.4f}', end=' ')
            print('\033[0m')            # reset

            if prev_rot is not None and rot_diff(prev_rot, rot) > np.pi/16:
                rot_list = discrete_rotation(prev_rot, rot)
                for R in rot_list:
                    robot.send_tcp_pose(xyz, R)
            else:
                robot.send_tcp_pose(xyz, rot)
            prev_rot = rot

            if prev_width is None or abs(prev_width-width) > 0.006 or (prev_width > 0 and width == 0):
                prev_width = width
                # TODO: blocking?
                gripper.set_width(width)
            end_time = time.time()
            print(f'[Main] Step time: {end_time - start_time:.4f}')
            time.sleep(max(0, step_time - (end_time - start_time)))
    finally:
        # clean up
        del agent
        for cam in global_cams:
            del cam
        for cam in wrist_cams:
            del cam
        del gripper

        # go back to ready pose before exiting
        xyz, rot = robot.get_tcp_pose()
        xyz[2] = 0.17
        robot.send_tcp_pose(xyz, rot)
        time.sleep(0.5)
        robot.send_tcp_pose(ready_xyz, ready_rot)
        time.sleep(1)
        del robot


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='name (without extension) of the config file')
    parser.add_argument('--ckpt', type=str, required=True, help='path to the model checkpoint file')

    parser.add_argument('--device', type=int, default=0, help='GPU device index')
    parser.add_argument('--seed', type=int, default=42, help='random seed for inference reproducibility')

    parser.add_argument('--max_timestep', type=int, default=1000, help='maximum number of timesteps to run the agent')
    parser.add_argument('--denoising_steps', type=int, default=16, help='number of denoising steps during inference')
    parser.add_argument('--act_horizon_eval', type=int, default=12, help='number of actions to use for evaluation')
    
    parser.add_argument('--ctrl_freq', type=int, default=10, help='control frequency of the robot')
    parser.add_argument('--pred_interval', type=int, default=4, help='predict new actions every pred_interval steps (if non-blocking, make sure the interval is longer than the prediction time)')

    parser.add_argument('--t_ensemble', action='store_true', help='use temporal ensemble for stable and smooth trajectory')

    main(parser.parse_args())
