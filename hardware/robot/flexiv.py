import time

import numpy as np
from transforms3d.quaternions import mat2quat, quat2mat

from hardware.robot.generic import Robot


class FlexivRobot(Robot):
    """
    Flexiv Robot Control Class
    """
    def __init__(self, robot_ip_address, pc_ip_address, step_sync=None, streaming_freq=10, waiting_gap=0.02):
        """
        Parameters:
            robot_ip_address: str
                ip address of the robot
            pc_ip_address: str
                ip address of the pc
            step_sync: multiprocessing.Barrier (default None)
                Barrier for step synchronization
            streaming_freq: int (default 10)
                Gripper streaming frequency if step_sync is None
            waiting_gap: float (default 0.02)
                Minimal interval between serial communications
        """
        # Download the Flexiv RDK and put the .so file here
        # https://rdk.flexiv.com/manual/getting_started.html
        from . import flexivrdk
        
        self.mode = flexivrdk.Mode
        self.mode_mapper = {
            'idle': 'IDLE',
            'primitive': 'NRT_PRIMITIVE_EXECUTION',
            'cart_impedance_online': 'NRT_CARTESIAN_MOTION_FORCE'
        }

        self.robot_states = flexivrdk.RobotStates()
        self.robot = flexivrdk.Robot(robot_ip_address, pc_ip_address)

        super().__init__(step_sync, streaming_freq, waiting_gap)

    def __del__(self):
        self.robot.stop()
        while self.robot.getMode() != self.mode_mapper['idle']:
            time.sleep(0.005)

    def initialize(self):
        # Clear fault on robot server if any
        if self.is_fault():
            print("[Robot] Fault occurred on robot server, trying to clear ...")
            # Try to clear the fault
            self.clear_fault()
            time.sleep(2)
            # Check again
            if self.is_fault():
                print("[Robot] Fault cannot be cleared, exiting ...")
                return
            print("[Robot] Fault on robot server is cleared")

        # Enable the robot, make sure the E-stop is released before enabling
        print("[Robot] Enabling robot ...")
        self.robot.enable()
        # Wait for the robot to become operational
        seconds_waited = 0
        while not self.is_operational():
            time.sleep(1)
            seconds_waited += 1
            if seconds_waited == 10:
                print("[Robot] Still waiting for robot to become operational, please check that the robot 1) has no fault, 2) is in [Auto (remote)] mode.")
        print("[Robot] Robot is now operational")

    def _update_status(self):
        self.robot.getRobotStates(self.robot_states)

    def switch_mode(self, mode, sleep_time=0.01):
        """
        Switch to different control modes

        Parameters:
            mode: str 'idle' | 'cart_impedance_online'
            sleep_time: float
                sleep time to control mode switch time
        """
        mode = getattr(self.mode, self.mode_mapper[mode.lower()])
        idle_mode = getattr(self.mode, self.mode_mapper['idle'])
        if self.robot.getMode() == mode:
            return

        while self.robot.getMode() != idle_mode:
            self.robot.setMode(idle_mode)
            time.sleep(sleep_time)
        while self.robot.getMode() != mode:
            self.robot.setMode(mode)
            time.sleep(sleep_time)

        print(f'[Robot] Set mode: {str(self.robot.getMode())}')
        
    def send_tcp_pose(self, xyz, rot):
        self.switch_mode("cart_impedance_online")
        quat = mat2quat(rot)
        tcp = np.concatenate([xyz, quat])
        self.command_queue.put((self.robot.sendCartesianMotionForce, (tcp,)))
    
    def get_tcp_pose(self, real_time=False):
        with self.status_lock:
            if real_time:
                self._update_status()
            # xyz + quat
            tcp = np.array(self.robot_states.tcpPose)
        xyz = tcp[:3]
        rot = quat2mat(tcp[3:7])
        return xyz, rot

    def get_tcp_vel(self, real_time=False):
        """
        Get current robot's tool velocity in world frame.

        Returns:
            7-dim list consisting of (vx,vy,vz,vrw,vrx,vry,vrz)
        """
        with self.status_lock:
            if real_time:
                self._update_status()
            return np.array(self.robot_states.tcpVel)

    def get_joint_pos(self, real_time=False):
        """
        Get current joint value.

        Returns:
            7-dim numpy array of 7 joint position
        """
        with self.status_lock:
            if real_time:
                self._update_status()
            return np.array(self.robot_states.q)

    def get_joint_vel(self, real_time=False):
        """
        Get current joint velocity.

        Returns:
            7-dim numpy array of 7 joint velocity
        """
        with self.status_lock:
            if real_time:
                self._update_status()
            return np.array(self.robot_states.dq)

    def set_max_contact_wrench(self, max_wrench):
        """
        Set max contact wrench for robotic arm.

        Parameters:
            max_wrench: np.array of shape (6,)
                max moving force (fx,fy,fz,wx,wy,wz)
        """
        self.switch_mode("cart_impedance_online")
        self.robot.setMaxContactWrench(max_wrench)

    def cali_force_sensor(self, data_collection_time=0.2):
        """
        Calibrate force sensor.

        Parameters:
            data_collection_time: float
                time for calibration
        """
        self.switch_mode('primitive')
        self.robot.executePrimitive(
            f'CaliForceSensor(dataCollectionTime={data_collection_time})'
        )

    def clear_fault(self):
        self.robot.clearFault()

    def is_fault(self):
        """
        Check if robot is in FAULT state.
        """
        return self.robot.isFault()

    def is_operational(self):
        return self.robot.isOperational()
