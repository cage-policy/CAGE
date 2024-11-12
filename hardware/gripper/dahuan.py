import time

import modbus_tk.defines as cst
import serial
from modbus_tk import modbus_rtu

from hardware.gripper.generic import Gripper


class DahuanAG95Gripper(Gripper):
    """
    Dahuan AG-95 Gripper Control Class
        real-world width range: 0mm ~ 95mm.
        real-world force range: 45N ~ 160N.
    """
    def __init__(self, port, step_sync=None, streaming_freq=10, waiting_gap=0.01):
        """
        Parameters:
            port: str (e.g. '/dev/ttyUSB0')
                Serial port for Dahuan gripper
            step_sync: multiprocessing.Barrier (default None)
                Barrier for step synchronization
            streaming_freq: int (default 10)
                Gripper streaming frequency if step_sync is None
            waiting_gap: float (default 0.01)
                Minimal interval between serial communications
        """
        self.master = modbus_rtu.RtuMaster(
            serial.Serial(
                port=port, baudrate=115200, bytesize=8, parity="N", stopbits=1
            )
        )
        self.master.open()
        assert self.master._is_opened, "[Gripper] Port {} needs permission".format(port)
        self.master.set_timeout(1.0)
        self.master.set_verbose(True)

        self.max_width = 0.095

        super().__init__(step_sync, streaming_freq, waiting_gap)

    def init_open(self):
        self.master.execute(1, cst.WRITE_SINGLE_REGISTER, 0x0100, 2, 0x0001)
        return_data = self.master.execute(1, cst.READ_HOLDING_REGISTERS, 0x0200, 1)
        while return_data != 1:
            return_data = self.master.execute(1, cst.READ_HOLDING_REGISTERS, 0x0200, 1)[0]
            time.sleep(0.1)

    def is_idle(self):
        with self.info_lock:
            self._update_info()
            return self.info[2] != 0

    def _update_info(self):
        """
        Get the current gripper information, including width, current and status
            width: real-world width (in meters) of the gripper
            status:
                0 : moving
                1 : stop and reach target position
                2 : stop and catch object
                3 : cathed object and then object is fallen
        """
        with self.serial_lock:
            width = self.master.execute(1, cst.READ_HOLDING_REGISTERS, 0x0202, 1)
            current = self.master.execute(1, cst.READ_HOLDING_REGISTERS, 0x0204, 1)
            status = self.master.execute(1, cst.READ_HOLDING_REGISTERS, 0x0201, 1)

        self.info = [width[0]/1000 * self.max_width, current[0], status[0]]
    
    def _set_force(self, force_percent):
        force = round(force_percent * 100)
        if force <= 20:
            force = 21
        with self.serial_lock:
            return_data = self.master.execute(
                1, cst.WRITE_SINGLE_REGISTER, 0x0101, 2, force
            )[1]
        assert return_data == force, 'Set force returned with unexpected value'
    
    def _set_width(self, width):
        width_permillage = round(width/self.max_width * 1000)
        with self.serial_lock:
            return_data = self.master.execute(
                1, cst.WRITE_SINGLE_REGISTER, 0x0103, 2, width_permillage
            )[1]
        assert return_data == width_permillage, 'Set width returned with unexpected value'
