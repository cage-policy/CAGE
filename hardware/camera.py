import time
from multiprocessing import Event, Lock, Process, Value
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import pyrealsense2 as rs

FRAME_RATE = 30
RESOLUTION = (1280, 720)


class RealSenseRGBDCamera:
    """
    Intel RealSense D435/D415 RGB-D Camera Streamer
    """
    def __init__(self, serial, enable_depth=False, step_sync=None, streaming_freq=FRAME_RATE):
        """
        Parameters:
            serial: str
                Camera serial number
            enable_depth: bool (default False)
                Whether to enable depth streaming
            step_sync: multiprocessing.Barrier (default None)
                Barrier for step synchronization
            streaming_freq: int (default FRAME_RATE)
                Camera streaming frequency if step_sync is None
        """
        self.enable_depth = enable_depth

        temp = np.zeros((RESOLUTION[1], RESOLUTION[0], 3), dtype=np.uint8)
        self.shm_color = SharedMemory(name=f'{serial}_color', create=True, size=temp.nbytes)
        self.rgb_raw = np.ndarray(temp.shape, dtype=temp.dtype, buffer=self.shm_color.buf)
        if enable_depth:
            temp = np.zeros((RESOLUTION[1], RESOLUTION[0]), dtype=np.float32)
            self.shm_depth = SharedMemory(name=f'{serial}_depth', create=True, size=temp.nbytes)
            self.depth_raw = np.ndarray(temp.shape, dtype=temp.dtype, buffer=self.shm_depth.buf)

        self.timer = None
        self.step_interval = None
        if step_sync is None:
            self.timer = Event()
            self.step_interval = 1.0 / streaming_freq
        
        self.streaming = Value('i', 0)
        self.lock = Lock()
        self.process = Process(target=self._streaming,
                                args=(serial, enable_depth, self.streaming, self.lock, step_sync or self.timer, self.step_interval),
                                daemon=True)
        self.process.start()
        # ensure the camera is ready, sometimes it takes quite a while to initialize
        while self.streaming.value != 1:
            time.sleep(0.1)
        
    def __del__(self):
        self.streaming.value = 0
        if self.timer is not None:
            self.timer.set()
        else:
            self.process.terminate()
        self.process.join()

        self.shm_color.close()
        self.shm_color.unlink()
        if self.enable_depth:
            self.shm_depth.close()
            self.shm_depth.unlink()

    def get_observation(self):
        """
        Get the latest observation from the camera

        Returns:
            (rgb, [depth]): numpy array
                RGB or RGB-D image (if depth is enabled)
        """
        with self.lock:
            rgb = self.rgb_raw.copy()
            if self.enable_depth:
                depth = self.depth_raw.copy()
        if self.enable_depth:
            return rgb, depth
        return rgb

    @staticmethod
    def _streaming(serial, enable_depth, streaming, lock, timer, step_interval):
        pipeline = rs.pipeline()
        config = rs.config()

        config.enable_device(serial)
        config.enable_stream(rs.stream.color, RESOLUTION[0], RESOLUTION[1], rs.format.rgb8, FRAME_RATE)
        if enable_depth:
            config.enable_stream(rs.stream.depth, RESOLUTION[0], RESOLUTION[1], rs.format.z16, FRAME_RATE)
        align = rs.align(rs.stream.color)

        pipeline.start(config)

        # drop first few frames
        for _ in range(40):
            frameset = align.process(pipeline.wait_for_frames())
            color_image = np.asanyarray(frameset.get_color_frame().get_data()).astype(np.uint8)
            if enable_depth:
                depth_image = np.asanyarray(frameset.get_depth_frame().get_data()).astype(np.float32) / 1000.

        shm_color = SharedMemory(name=f'{serial}_color')
        shm_color_buf = np.ndarray(color_image.shape, dtype=color_image.dtype, buffer=shm_color.buf)
        if enable_depth:
            shm_depth = SharedMemory(name=f'{serial}_depth')
            shm_depth_buf = np.ndarray(depth_image.shape, dtype=depth_image.dtype, buffer=shm_depth.buf)
        streaming.value = 1
        
        print(f'[Camera {serial}] Start streaming ...')
        try:
            while streaming.value == 1:
                with lock:
                    frameset = align.process(pipeline.wait_for_frames())
                    color_image = np.asanyarray(frameset.get_color_frame().get_data()).astype(np.uint8)
                    if enable_depth:
                        depth_image = np.asanyarray(frameset.get_depth_frame().get_data()).astype(np.float32) / 1000.
                    
                    shm_color_buf[:] = color_image
                    if enable_depth:
                        shm_depth_buf[:] = depth_image

                timer.wait(step_interval)
        finally:
            shm_color.close()
            if enable_depth:
                shm_depth.close()
            pipeline.stop()
