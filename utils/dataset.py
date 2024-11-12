import os
import shutil
import time

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.v2 as T
from accelerate.logging import get_logger
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.constants import TRANS_MAX, TRANS_MIN
from utils.preprocessings import (camera_orientation_correction,
                                  decode_gripper_width, get_normalized_actions,
                                  tcp_to_xyz_rot)

logger = get_logger(__name__)

# camera infos for RH20T
IN_HAND_CAMS = {
    1: ['cam_043322070878'],
    2: ['cam_104422070042'],
    3: ['cam_045322071843'],
    4: ['cam_045322071843'],
    5: ['cam_104422070042', 'cam_135122079702'],
    6: ['cam_135122070361', 'cam_135122075425'],
    7: ['cam_135122070361', 'cam_135122075425']
}

FRONT_CAM = {
    1: ['cam_035622060973','cam_039422060546','cam_750612070851','cam_750612070853'],
    2: ['cam_037522062165','cam_104122061850','cam_f0461559'],
    3: ['cam_038522062288','cam_104122062295','cam_104422070011','cam_f0172289'],
    4: ['cam_038522062288','cam_104122062295','cam_104422070011','cam_f0172289'],
    5: ['cam_037522062165','cam_104122061850','cam_f0461559'],
    6: ['cam_104122061018','cam_104122061330','cam_f0271510'],
    7: ['cam_104122061018','cam_104122061330','cam_f0271510'],
}
SIDE_CAM = {
    1: ['cam_038522062547'],
    2: ['cam_104122063678','cam_105422061350'],
    3: ['cam_104122063550'],
    4: ['cam_104122063550'],
    5: ['cam_104122063678','cam_105422061350'],
    6: ['cam_037522061512','cam_104122063633','cam_104122064161'],
    7: ['cam_037522061512','cam_104122063633','cam_104122064161'],
}
BACK_CAM = {
    1: ['cam_819112070044'],
    2: ['cam_036422060215','cam_104422070044'],
    3: ['cam_036422060909','cam_104122062823'],
    4: ['cam_036422060909','cam_104122062823'],
    5: ['cam_036422060215','cam_104422070044'],
    6: ['cam_104122060811','cam_104122061602'],
    7: ['cam_104122060811','cam_104122061602'],
}
type2cams = {
    'front': FRONT_CAM,
    'side': SIDE_CAM,
    'back': BACK_CAM,
    'all': FRONT_CAM | SIDE_CAM | BACK_CAM,
}

CACHE_DIR = './cache'
MAX_CACHE = 50


class RealWorldDataset(Dataset):
    """
    Realworld demonstration dataset following RH20T format
    """
    def __init__(self, obs_horizon, act_horizon, act_repr,
                 meta_data,
                 sample_interval=1, timestamp_unit=1000,
                 augment=False, augment_color=False,
                 img_size=224,
                 save_cache=True):
        """
        Parameters
            obs_horizon: int
                Horizon of latest observations
            act_horizon: int
                Horizon of future action sequence to predict
            act_repr: 'abs' | 'rel' | 'delta_xyz'
                Representation of actions
                    abs: xyz, rot, width absolute
                    rel: xyz & rot relative, width absolute
                    delta_xyz: xyz relative, rot & width absolute
            meta_data: dict
                Config of specific dataset, used to load samples
                    e.g. root_path, selected_tasks, camera_ids, etc.
            sample_interval: int (default: 1)
                Interval between two consecutive frames
            timestamp_unit: int (default: 1000)
                Define the timestamp unit from which is considered as a sample
            augment: bool (default: False)
                Whether to apply geometric camera augmentation
            augment_color: bool (default: False)
                Whether to apply global color augmentation
        """
        super().__init__()

        self.main_cam = meta_data['camera_ids']['main']
        self.use_in_hand = meta_data['in_hand_views'] != 0
        self.use_fixed = meta_data['fixed_views'] != 0
        # number of in-hand cameras to use
        self.in_hand_views = meta_data['in_hand_views']
        # number of fixed cameras to use
        self.fixed_views = meta_data['fixed_views']

        assert self.main_cam in ['in_hand', 'fixed'], 'Main camera should be either in_hand or fixed'
        assert (self.main_cam == 'in_hand' and self.use_in_hand) or (self.main_cam == 'fixed' and self.use_fixed), 'Main camera should be used!'

        # split: 'train' | 'val' | (0, 1) | int | None
        #   train/val - reserve one user per config for validation
        #   (0, 1) - ratio of data to use (>=0.5 starts from the beginning, <0.5 starts from the end)
        #   int - number of samples to use
        #   None - do not split
        if meta_data['split'] is not None:
            assert meta_data['split'] in ['train', 'val'] or (0 < meta_data['split'] < 1) or (isinstance(meta_data['split'], int) and meta_data['split'] >= 1), "split should be one of None, 'train'|'val', a ratio between 0 and 1 or a positive integer"
        assert act_repr in ['abs', 'rel', 'delta_xyz'], f'Invalid action representation {act_repr}'

        self.sample_interval = sample_interval
        self.timestamp_unit = timestamp_unit

        self.act_repr = act_repr
        self.act_horizon = act_horizon
        self.obs_horizon = obs_horizon

        self.augment = augment
        img_size_aug = int(img_size / 0.875)    # resize to 256 for aug
        # preprocessor - [global_aug] - img_aug/no_aug - normalize
        self.img_preprocessor = T.Compose([
            T.ToImage(),
            T.Resize(img_size_aug),
        ])
        
        self.global_augment = None  # for all cameras
        self.cam_augment = None
        self.no_augment = T.Compose([
            T.CenterCrop(img_size),
        ])

        self.normalize = T.Compose([
            T.ToDtype(torch.float32, scale=True), # convert to float and normalize to [0, 1]
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        if augment:
            # introduce randomness in viewpoint for each FIXED camera
            self.cam_augment = T.Compose([
                T.RandomPerspective(distortion_scale=0.2),
                # random size in [200, 256]
                T.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(1., 1.)),
            ])

        if augment_color:
            # same params as dinov2
            self.global_augment = T.RandomApply(
                [T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8,
            )

        self._filter_data(meta_data)

        # parameters used when loading samples
        cur_params = [self.main_cam, self.timestamp_unit]

        loaded = False
        os.makedirs(CACHE_DIR, exist_ok=True)
        cache_files = sorted(os.listdir(CACHE_DIR))
        for folder in cache_files:
            df = pd.read_csv(os.path.join(CACHE_DIR, folder, 'meta.csv'), sep=' ')
            params = np.load(os.path.join(CACHE_DIR, folder, 'meta.npy'), allow_pickle=True)
            if np.all(params==cur_params) and df.equals(self.data):
                arr = np.load(os.path.join(CACHE_DIR, folder, 'data.npz'), allow_pickle=True)
                self.task_indices = arr['arr_0']
                self.reference_timestamps = arr['arr_1']
                self.tasks = arr['arr_2']
                self.num_samples = len(self.task_indices)
                loaded = True
                break
        # if no cache file found, load data and save the cache files
        if not loaded:
            self._load_samples()

            # should only be called on main process
            if save_cache:
                # save loaded data to cache files
                if len(cache_files) >= MAX_CACHE:
                    # delete the oldest
                    shutil.rmtree(os.path.join(CACHE_DIR, cache_files[0]))
                # create new folder in 'cache' with current timestamp as name
                cache_dir = os.path.join(CACHE_DIR, f'{time.time():.2f}')
                os.makedirs(cache_dir)

                np.save(os.path.join(cache_dir, 'meta.npy'), cur_params)
                self.data.to_csv(os.path.join(cache_dir, 'meta.csv'), sep=' ', index=False)
                np.savez_compressed(os.path.join(cache_dir, 'data.npz'), self.task_indices, self.reference_timestamps, self.tasks)
        
        cnts_per_config = self.data.groupby('config').path.unique().apply(len)
        logger.info(f'Loaded {len(self)} samples {"from cache" if loaded else ""}')
        logger.info('with:')
        for idx in cnts_per_config.index:
            logger.info(f'  config {idx}: {cnts_per_config[idx]} demos')

    def __len__(self):
        return self.num_samples
    
    def _filter_data(self, meta_data):
        # load data from csv
        if isinstance(meta_data['path'], str):
            df = pd.read_csv(meta_data['path'], sep=' ')
        else:   # list of dataframes to use
            dfs = []
            for path in meta_data['path']:
                dfs.append(pd.read_csv(path, sep=' '))
            df = pd.concat(dfs)
            df.reset_index(drop=True, inplace=True)
        # keys: ['path', 'task', 'user', 'scene', 'config', 'cam_id', 'cam_type', 'start_timestamp', 'end_timestamp', 'start_frame_idx', 'end_frame_idx', 'start_frame_timestamp']

        selected_configs = meta_data['configs']
        selected_tasks = meta_data['tasks']
        # by default only use tasks with id < 200
        mask = True if selected_tasks is None else df.task.isin(selected_tasks)
        if selected_configs is not None:
            mask &= df.config.isin(selected_configs)
        df = df[mask]

        # == None: select among all available cameras
        # != None: select among given cameras
        # {1:[cam_id],2:[cam_id]...}
        in_hand_cam_ids = meta_data['camera_ids']['in_hand']
        if isinstance(meta_data['camera_ids']['fixed'], str):
            cam_types = meta_data['camera_ids']['fixed'].split('+')
            fixed_cam_ids = type2cams[cam_types[0]]
            for t in cam_types[1:]:
                fixed_cam_ids = {k: v+type2cams[t][k] for k, v in fixed_cam_ids.items()}
        else:   # list or None
            fixed_cam_ids = meta_data['camera_ids']['fixed']
        idx_to_keep = []
        for config, config_data in df.groupby('config'):
            mask = False
            if self.use_in_hand:
                in_hand_mask = config_data.cam_type == 'in_hand'
                if in_hand_cam_ids is not None and in_hand_cam_ids[config] is not None:
                    in_hand_mask &= config_data.cam_id.isin(in_hand_cam_ids[config])
                mask |= in_hand_mask
            if self.use_fixed:
                fixed_mask = config_data.cam_type == 'fixed'
                if fixed_cam_ids is not None and fixed_cam_ids[config] is not None:
                    fixed_mask &= config_data.cam_id.isin(fixed_cam_ids[config])
                mask |= fixed_mask
            idx_to_keep.extend(config_data[mask].index)
        df = df.loc[idx_to_keep]

        if meta_data['split'] == 'train' or meta_data['split'] == 'val':
            # take one user per config as validation set
            pct_per_user = df.groupby(['config','user']).path.count()/df.groupby('config').path.count()
            val_user_per_cfg = (pct_per_user-0.1).abs().groupby('config').idxmin()
            all_indices = pd.Series(zip(df.config, df.user), index=df.index)
            if meta_data['split'] == 'val':
                df = df[all_indices.isin(val_user_per_cfg)]
            else:
                df = df[~all_indices.isin(val_user_per_cfg)]
        elif meta_data['split'] is not None:   # ratio or num of samples
            # first shuffle (index)
            samples = df.path.unique()
            np.random.shuffle(samples)
            if 0.5 <= meta_data['split'] < 1:
                samples = samples[:int(meta_data['split']*len(samples))]
            elif 0 < meta_data['split'] < 0.5:
                samples = samples[-int(meta_data['split']*len(samples)):]
            else:
                samples = samples[:meta_data['split']]
            df = df[df.path.isin(samples)]

        idx_to_keep = []
        for _, task_data in df.groupby('path'):
            if (task_data.cam_type=='in_hand').sum() >= self.in_hand_views and (task_data.cam_type=='fixed').sum() >= self.fixed_views:
                idx_to_keep.extend(task_data.index)
        df = df.loc[idx_to_keep]
        
        df.reset_index(drop=True, inplace=True)
        self.data = df

    def _load_samples(self):
        self.task_indices = []
        self.reference_timestamps = []
        # each item is a dict containing task related data
        # i.e. info, in-hand cameras, fixed cameras...
        self.tasks = []
        for task_path, task_data in tqdm(self.data.groupby('path'), desc='Loading tasks...'):
            task_dict = {}

            s = task_data.iloc[0]
            task_dict['info'] = {
                'task': int(s.task),
                'user': int(s.user),
                'scene': int(s.scene),
                'config': int(s.config),
                'path': task_path,
            }

            # the length of a task is defined to be the number of the units in the common part of the video
            start_timestamp = s.start_timestamp
            end_timestamp = s.end_timestamp
            length = int(np.ceil((end_timestamp - start_timestamp) / self.timestamp_unit))
            cnts = np.zeros(length)

            task_dict['fixed'] = {}
            task_dict['in_hand'] = {}
            for _, row in task_data.iterrows():
                cam_id = row.cam_id

                # load timestamps
                timestamp_path = os.path.join(task_path, cam_id, 'timestamps.npy')
                if os.path.exists(timestamp_path):
                    timestamps = np.load(timestamp_path, allow_pickle=True).item()['color']
                else:
                    timestamps = sorted([int(t[:-4]) for t in os.listdir(os.path.join(task_path, cam_id, 'color'))])

                tcp_dir = os.path.join(task_path, cam_id, 'tcp')
                gripper_info_dir = os.path.join(task_path, cam_id, 'gripper_info')
                gripper_cmd_dir = os.path.join(task_path, cam_id, 'gripper_command')

                arr = []
                for i in range(row.start_frame_idx, row.end_frame_idx+1):
                    timestamp = timestamps[i]
                    if row.cam_type == self.main_cam:
                        cnts[(timestamp - start_timestamp) // self.timestamp_unit] += 1

                    tcp_path = os.path.join(tcp_dir, '%d.npy'%timestamp)
                    gripper_pose = np.load(tcp_path)
                    gripper_xyz, gripper_mat = tcp_to_xyz_rot(gripper_pose, row.config)

                    gripper_path = os.path.join(gripper_info_dir, '%d.npy'%timestamp)
                    gripper_state = np.load(gripper_path)
                    gripper_cur_width = decode_gripper_width(gripper_state[0], row.config)

                    gripper_path = os.path.join(gripper_cmd_dir, '%d.npy'%timestamp)
                    gripper_state = np.load(gripper_path)
                    gripper_command_width = decode_gripper_width(gripper_state[0], row.config)

                    # dim=16 in total
                    arr.append([i, timestamp, *gripper_xyz, *gripper_mat.flatten(), gripper_cur_width, gripper_command_width])
                # fix potential overflow tcp values (e.g. -9.432e+36)
                arr = np.array(arr)
                idx = np.argwhere(np.abs(arr[:, 2:]) > 2)
                for i in idx:
                    n = i[0]
                    m = i[1]+2
                    prev = arr[n-1]
                    next = arr[n+1]
                    # interpolate the missing value
                    arr[n, m] = prev[m] + (next[m] - prev[m]) / (next[1] - prev[1]) * (arr[n, 1] - prev[1])
                
                task_dict[row.cam_type][cam_id] = arr

            idx = np.flatnonzero(cnts)
            length = len(idx)
            self.task_indices.extend([len(self.tasks)] * length)
            self.reference_timestamps.extend([start_timestamp + i*self.timestamp_unit for i in idx])
            self.tasks.append(task_dict)

        self.num_samples = len(self.task_indices)

    def __getitem__(self, index):
        ret = {}

        task_index = self.task_indices[index]
        ref_timestamp = self.reference_timestamps[index]
        task = self.tasks[task_index]

        # 1. Observation
        # randomly choose which cameras to use
        cam_sampled = {}
        obs = {}
        if self.use_in_hand:
            cam_sampled['in_hand'] = np.random.choice(list(task['in_hand'].keys()), size=self.in_hand_views, replace=False).tolist()
            obs['in_hand'] = []
        if self.use_fixed:
            cam_sampled['fixed'] = np.random.choice(list(task['fixed'].keys()), size=self.fixed_views, replace=False).tolist()
            obs['fixed'] = []

        # extract observations from main camera first
        main_cam_id = cam_sampled[self.main_cam].pop(0)
        ret['main_cam'] = main_cam_id

        main_cam_data = task[self.main_cam][main_cam_id]
        ts_diff = main_cam_data[:, 1] - ref_timestamp
        candidate_indices = np.flatnonzero((ts_diff >= 0) & (ts_diff < self.timestamp_unit))
        if len(candidate_indices) > 0:
            sample_index = np.random.choice(candidate_indices)
        else:
            sample_index = np.argmin(np.abs(ts_diff))
        idx = np.arange(
            sample_index - self.sample_interval * (self.obs_horizon-1),
            sample_index+1,
            step = self.sample_interval,
        ).clip(min=0)
        obs_timestamps, obs_tcp = main_cam_data[idx, 1], main_cam_data[idx, 2:]
        obs_timestamps = obs_timestamps.astype(int)

        obs[self.main_cam].append(self._get_frames(task['info'], main_cam_id, obs_timestamps))

        # load frames from other cameras
        for cam_type, cam_ids in cam_sampled.items():
            for cam_id in cam_ids:
                cam_data = task[cam_type][cam_id]
                idx = self._get_closest_indices(cam_data[:, 1], obs_timestamps)
                obs[cam_type].append(self._get_frames(task['info'], cam_id, cam_data[idx, 1].astype(int)))

        # last frame: timestamp = t
        ret['obs_timestamps'] = obs_timestamps
        # base pose for the calculation of relative actions
        obs_pose = np.zeros((4, 4))
        obs_pose[:3, 3] = (obs_tcp[-1, :3] - TRANS_MIN) / (TRANS_MAX - TRANS_MIN)
        obs_pose[:3, :3] = obs_tcp[-1, 3:12].reshape(3, 3)
        obs_pose[3, 3] = 1
        ret['obs_pose'] = obs_pose

        # apply global augmentation
        if self.global_augment is not None:
            all_obs = []
            for _, cam_obs in obs.items():
                for cam_obs_i in cam_obs:
                    all_obs.append(cam_obs_i)
            all_obs = torch.cat(all_obs)  # N * obs_horizon, C, H, W
            # (IMPORTANT) apply the SAME global augmentations to all cameras
            all_obs = self.global_augment(all_obs)
            # split back
            idx = 0
            for _, cam_obs in obs.items():
                for i in range(len(cam_obs)):
                    cam_obs[i] = all_obs[idx*self.obs_horizon: (idx+1)*self.obs_horizon]
                    idx += 1

        # apply camera-specific augmentation
        for cam_type, cam_obs in obs.items():
            if self.cam_augment is not None and cam_type == 'fixed':
                for i in range(len(cam_obs)):
                    cam_obs[i] = self.cam_augment(cam_obs[i])
            else:
                for i in range(len(cam_obs)):
                    cam_obs[i] = self.no_augment(cam_obs[i])

            # in_hand/fixed_views, obs_window, C, H, W
            ret[cam_type+'_obs'] = self.normalize(torch.stack(cam_obs))

        # 2. Proprioceptions
        # uses actual width
        ret['proprio'] = get_normalized_actions(obs_tcp[:, :13], obs_pose, self.act_repr)

        # 3. Action
        idx = np.arange(
            sample_index,
            sample_index + self.sample_interval * (self.act_horizon+1),
            step = self.sample_interval,
        )
        padding_mask = idx >= len(main_cam_data)
        idx = idx.clip(max=len(main_cam_data)-1)
        act_timestamps, tcp = main_cam_data[idx, 1], main_cam_data[idx, 2:]

        # first frame: next timestamp t+1
        ret['action_timestamps'] = act_timestamps[1:].astype(int)
        ret['action_padding_mask'] = padding_mask
        # use gripper_command as width
        tcp = tcp[1:]
        tcp = np.concatenate([tcp[:, :12], tcp[:, 13:]], axis=1)
        ret['actions'] = get_normalized_actions(tcp, obs_pose, self.act_repr)

        return ret
    
    def _get_frames(self, task_info, cam_id, frame_timestamps):
        img_folder = os.path.join(task_info['path'], cam_id, 'color')
        imgs = []
        for timestamp in frame_timestamps:
            img_path = os.path.join(img_folder, f'{timestamp}.png')
            img = Image.open(img_path)
            img = camera_orientation_correction(img, cam_id, task_info['config'], task_info['task'], task_info['user'])
            imgs.append(self.img_preprocessor(img))
        imgs = torch.stack(imgs)    # N, C, H, W

        return imgs
    
    def _get_closest_indices(self, timestamps, ref_timestamps):
        indices = []
        ref_i = 0
        i = 0
        while i < len(timestamps):
            if timestamps[i] >= ref_timestamps[ref_i]:
                if i == 0 or timestamps[i] - ref_timestamps[ref_i] < ref_timestamps[ref_i] - timestamps[i-1]:
                    indices.append(i)
                else:
                    indices.append(i-1)
                ref_i += 1
                if ref_i == len(ref_timestamps):
                    break
            else:
                i += 1
        while len(indices) < len(ref_timestamps):
            indices.append(len(timestamps)-1)
        return indices
