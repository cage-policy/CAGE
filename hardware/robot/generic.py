import time
from queue import Queue
from threading import Event, Lock, Thread


class Robot:
    """
    Generic Robot Control Class
        with 2 sub-threads for streaming and command execution
    """
    def __init__(self, step_sync=None, streaming_freq=10, waiting_gap=0.02):
        """
        Parameters:
            step_sync: multiprocessing.Barrier (default None)
                Barrier for step synchronization
            streaming_freq: int (default 10)
                Gripper streaming frequency if step_sync is None
            waiting_gap: float (default 0.02)
                Minimal interval between serial communications
        """
        self.timer = step_sync or Event()
        self.step_interval = None if step_sync is not None else 1.0 / streaming_freq
        self.waiting_gap = waiting_gap

        self.status_lock = Lock()
        self.command_lock = Lock()

        self.command_queue = Queue()
        self.command_thread = Thread(
            target=self._command_execution,
            daemon=True)
        self.command_thread.start()

        self.initialize()

        self.is_streaming = False
        self.streaming_thread = Thread(
            target=self._streaming,
            daemon=True)
        self.streaming_thread.start()

    def __del__(self):
        self.is_streaming = False
        if isinstance(self.timer, Event):
            self.timer.set()
        else:
            self.timer.abort()
        self.streaming_thread.join()

        self.command_queue.put(None)
        self.command_thread.join()

    def _streaming(self):
        print('[Robot] Start streaming ...')
        self.is_streaming = True
        while self.is_streaming:
            try:
                with self.status_lock:
                    self._update_status()
            except Exception as e:
                print("[Robot] Error in streamer: ", e)
            self.timer.wait(self.step_interval)

    def _command_execution(self):
        while True:
            cmd = self.command_queue.get()
            if cmd is None:
                break
            
            func, args = cmd
            try:
                func(*args)
            except Exception as e:
                print("[Robot] Error in command execution: ", e)
            time.sleep(self.waiting_gap)

    def initialize(self):
        """
        Initialize the robot
        """
        raise NotImplementedError
    
    def _update_status(self):
        """
        Update robot status (tcp, joint, etc.)
        """
        raise NotImplementedError
    
    def send_tcp_pose(self, xyz, rot):
        """
        Send target pose to robot

        Parameters:
            xyz: numpy array of shape (3,)
                target position
            rot: numpy array of shape (3, 3)
                target rotation matrix
        """
        raise NotImplementedError
    
    def get_tcp_pose(self, real_time=False):
        """
        Get current robot's pose in world frame.

        Parameters:
            real_time: bool (default False)
                Whether to update status before getting pose

        Returns:
            (xyz, rot): numpy array of shape (3,) and (3, 3)
                current position and rotation matrix
        """
        raise NotImplementedError
