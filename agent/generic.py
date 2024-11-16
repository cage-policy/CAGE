import time
from multiprocessing import Array, Event, Lock, Process
from multiprocessing.shared_memory import SharedMemory

import numpy as np

from utils.constants import ACTION_DIM
from utils.transforms import rot6d_to_rotation_matrix


class Agent:
    """
    Generic Embodied Agent
        The inference interface for real-world evaluations.
    """
    def __init__(self, configs, 
                 blocking=True,
                 temporal_ensemble=True, k=0.01):
        """
        Parameters:
            configs: omegaconf.DictConfig
                policy-specific configs and command-line args.
            blocking: bool
                whether wait for the completion of inference when calling `update_actions`.
            temporal_ensemble: bool
                whether to apply action chunking for predicted actions.
                ref: [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](http://arxiv.org/abs/2304.13705)
            k: float | List[float|None]
                the exponential coefficient of temporal_ensemble for translation, rotation and width.
                if k is list, then translation, rotation and width will each have a different smoothing coefficient. None means no smoothing.
        """
        self.configs = configs
        self.blocking = blocking

        self.temporal_ensemble = temporal_ensemble
        self.k = None
        if self.temporal_ensemble:
            if isinstance(k, float):
                self.k = [k, k, k]
            else:
                assert k is not None and len(k) == 3, 'k should be a list of 3 floats!'
                self.k = k

        # action buffer for temporal ensemble
        max_timestep = configs.max_timestep
        act_horizon = configs.act_horizon_eval
        temp = np.zeros((max_timestep, max_timestep+act_horizon, ACTION_DIM), dtype=np.float32)
        self.act_shm = SharedMemory(name='action_buffer', create=True, size=temp.nbytes)
        self.action_buffer = np.ndarray(temp.shape, dtype=temp.dtype, buffer=self.act_shm.buf)
        self.action_buffer[:] = temp
        self.act_horizon = act_horizon

        # start the inference process
        self.shared_status = Array('i', [0, 0, 0]) # ready, cur_step, update_step
        self.obs_lock = Lock()
        self.act_lock = Lock()
        self.new_obs = Event()
        self.pred_done = Event()
        self.policy_process = Process(target=self._policy_runner,
                                      args=(configs, self.shared_status, self.obs_lock, self.act_lock, self.new_obs, self.pred_done),
                                      daemon=True)
        self.policy_process.start()
        # wait for the initialization to complete
        while self.shared_status[0] != 1:
            time.sleep(0.5)

    def __del__(self):
        self.policy_process.terminate()
        self.policy_process.join()
        
        self.act_shm.close()
        self.act_shm.unlink()

    @staticmethod
    def _policy_runner(configs, shared_status, obs_lock, act_lock, new_obs, pred_done):
        """
        The policy inference process. action_buffer should be updated iteratively based on new observations.

        Parameters:
            configs: omegaconf.DictConfig
                policy-specific configs and command-line args.
            shared_satatus: Array of [ready, cur_step, update_step]
                shared status between the main and the inference processes.
            obs_lock: Lock
                read-write lock for the observation buffers.
            act_lock: Lock
                read-write lock for the action buffer.
            new_obs: Event
                signal new observations.
            pred_done: Event
                signal the completion of one prediction.
        """
        raise NotImplementedError

    def _update_observation(self, obs):
        """
        Update the observations given the observation dict.

        Parameters:
            obs: dict
                observations from the environment.
        """
        raise NotImplementedError
    
    def update_actions(self, obs, timestep=0, reset=False):
        """
        Update the action buffer given the observation dict and current timestep.
        
        Parameters:
            obs: dict
                observations from the environment.
            timestep: int
                current timestep.
            reset: bool
                whether to reset the update_step to cur_step.
        """
        with self.obs_lock:
            # obs is handled by child class's implementation
            self._update_observation(obs)

            # update the current timestep
            self.shared_status[1] = timestep
            if reset or self.shared_status[2] < timestep:
                self.shared_status[2] = timestep
        
        # signal new observations are available
        self.new_obs.set()

        if self.blocking:
            # wait for the prediction to be done
            self.pred_done.wait()
            self.pred_done.clear()
            # if new obs not handled yet (update happens during the prediction)
            if self.new_obs.is_set():
                # wait again
                self.pred_done.wait()
                self.pred_done.clear()

    def get_action(self, timestep=0, blocking=True):
        """
        Get the target action at given timestep.
        
        Parameters:
            timestep: int
                the timestep for the action.
            blocking: bool
                whether to block the thread until the action is available.
        
        Returns:
            (xyz, M, w)
                7-Dof end-effector action represented in translation (xyz), rotation matrix (M) and width (w).
            or None if the action is not available and blocking=False.
        """
        self.act_lock.acquire()
        # if no action available
        if np.all(self.action_buffer[timestep, timestep]==0):
            self.act_lock.release()

            if not blocking:
                return None

            # wait for the prediction to be done
            self.pred_done.wait()
            self.pred_done.clear()

            self.act_lock.acquire()
        if self.temporal_ensemble:
            actions_for_curr_step = self.action_buffer[:timestep+1, timestep].copy()
            actions_populated = np.any(actions_for_curr_step != 0, axis=1)
            actions_for_curr_step = actions_for_curr_step[actions_populated]
            
            # coef for each dim
            weights = np.zeros((len(actions_for_curr_step), ACTION_DIM))
            weights[-1, :] = 1  # default to no smoothing

            if self.k[0] is not None:   # translation
                weights[:, :3] = np.exp(-self.k[0] * np.arange(len(actions_for_curr_step))[:, None])
            if self.k[1] is not None:   # rotation
                weights[:, 3:9] = np.exp(-self.k[1] * np.arange(len(actions_for_curr_step))[:, None])
            if self.k[2] is not None:   # width
                weights[:, 9] = np.exp(-self.k[2] * np.arange(len(actions_for_curr_step)))

            weights = weights / weights.sum(axis=0, keepdims=True)
            raw_action = (actions_for_curr_step * weights).sum(axis=0)
        else:
            raw_action = self.action_buffer[timestep, timestep].copy()
        # current action is scheduled to execute
        # increase the update_step
        if self.shared_status[2] < timestep + 1:
            self.shared_status[2] = timestep + 1
        self.act_lock.release()

        xyz = raw_action[:3]
        M = rot6d_to_rotation_matrix(raw_action[3:9])
        w = raw_action[9]

        return xyz, M, w

    def __call__(self, obs, timestep=0, act_horizon=4):
        """
        Update the action buffer and get predicted action sequence in a blocking manner.

        Parameters:
            obs: dict
                observations from the environment.
            timestep: int
                current timestep.
            act_horizon: int
                length of the action sequence.
            
        Returns:
            (xyz, M, w)
                the predicted action sequence.
        """
        assert act_horizon <= self.act_horizon, 'the length of the returned action sequence should be less than or equal to the action horizon of one prediction!'

        # update the action buffer (blocking)
        block_t = self.blocking
        self.blocking = True
        self.update_actions(obs, timestep, reset=True)
        self.blocking = block_t

        # get the action sequence
        xyzs, Ms, ws = [], [], []
        for i in range(act_horizon):
            xyz, M, w = self.get_action(timestep+i)
            xyzs.append(xyz)
            Ms.append(M)
            ws.append(w)

        xyzs = np.stack(xyzs, axis=0)
        Ms = np.stack(Ms, axis=0)
        ws = np.stack(ws, axis=0)
        return xyzs, Ms, ws
