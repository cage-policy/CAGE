import time
from threading import Event, Lock, Thread


class Gripper:
    """
    Generic Gripper Control Class
        with a sub-thread for non-blocking streaming
    """
    def __init__(self, step_sync=None, streaming_freq=10, waiting_gap=0.01):
        """
        Parameters:
            step_sync: multiprocessing.Barrier (default None)
                Barrier for step synchronization
            streaming_freq: int (default 10)
                Gripper streaming frequency if step_sync is None
            waiting_gap: float (default 0.01)
                Minimal interval between serial communications
        """
        self.timer = step_sync or Event()
        self.step_interval = None if step_sync is not None else 1.0 / streaming_freq
        self.waiting_gap = waiting_gap

        # first element is always the real-world width (in meters)
        self.info = None
        self.info_lock = Lock()
        self.serial_lock = Lock()

        # initialize the gripper
        self.init_open()

        # start the streaming thread
        self.streaming_thread = Thread(
            target=self._streaming,
            daemon=True)
        self.streaming_thread.start()
    
    def _streaming(self):
        print('[Gripper] Start streaming ...')
        while True:
            try:
                with self.info_lock:
                    self._update_info()
            except Exception as e:
                print("[Gripper] Error in streamer: ", e)
            self.timer.wait(self.step_interval)

    def get_info(self):
        """
        Get the current gripper information. 
        
        Returns:
            info: list
                The first element is always the real-world width (in meters), the rest depends on the different grippers.
        """
        with self.info_lock:
            return self.info[:]

    def set_force(self, force_percent):
        """
        Set the gripper force.
        
        Parameters:
            force_percent: [0.2, 1]
                actual force ranges are different for different grippers
        """
        assert 0.2 <= force_percent <= 1, "Force percentage should be in [0.2, 1]"
        self._set_force(force_percent)

    def set_width(self, width, blocking=False):
        """
        Set the gripper width

        Parameters:
            width: float
                real-world width (in meters) of the gripper
            blocking: bool (default False)
                whether to wait until the gripper reaches target width or catches an object
        """
        width = max(0, min(width, self.max_width))
        while True:
            try:
                self._set_width(width)
                if blocking:
                    # the gripper might not reach the target width if catches an object
                    # in this case, wait until idle
                    while not self.is_idle() and abs(self.info[0] - width) >= 0.005:
                        time.sleep(0.05)
                break
            except Exception as e:
                print('[Gripper] Error in set_width: ', e)
                time.sleep(self.waiting_gap)

    def open_gripper(self, blocking=True):
        """
        Open the gripper to the maximum width
        
        Parameters:
            blocking: bool (default True)
                whether to wait until the command is completed
        """
        self.set_width(self.max_width, blocking)

    def close_gripper(self, blocking=True):
        """
        Close the gripper

        Parameters:
            blocking: bool (default True)
                whether to wait until the command is completed
        """
        self.set_width(0, blocking)
    
    def init_open(self):
        """
        Initialize and open the gripper
        """
        raise NotImplementedError

    def is_idle(self):
        """
        Check if the gripper is idle
        """
        raise NotImplementedError
    
    def _update_info(self):
        raise NotImplementedError
    
    def _set_force(self, force_percent):
        """
        Actual set_force implementation
        """
        raise NotImplementedError
    
    def _set_width(self, width):
        """
        Actual set_width implementation
        """
        raise NotImplementedError
