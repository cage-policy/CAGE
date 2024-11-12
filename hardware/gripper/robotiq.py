import struct
import time

import serial

from hardware.gripper.generic import Gripper


class Robotiq2F85Gripper(Gripper):
    """
    Robotiq 2F-85 Gripper Control Class
        real-world width range: 0mm ~ 85mm.
        real-world force range: 25N ~ 155N.

    Refererences:
        [1] https://assets.robotiq.com/website-assets/support_documents/document/2F-85_2F-140_Instruction_Manual_e-Series_PDF_20190206.pdf
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
        self.ser = serial.Serial(
            port=port, baudrate=115200, timeout=1,
            parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS
        )

        self.max_width = 0.085
        self._force = 77     # [0, 255] default 30% of the force
        self._speed = 100    # [0, 255]

        self.last_write_time = time.time()

        super().__init__(step_sync, streaming_freq, waiting_gap)

    def init_open(self):
        self._activate()
        self.open_gripper()

    def is_idle(self):
        with self.info_lock:
            self._update_info()
            return self.info[2] in [0, 1]

    def set_speed(self, speed_percent):
        """
        Set the gripper action speed.
        
        Parameters:
            speed_percent: [0.2, 1]
        """
        assert 0.2 <= speed_percent <= 1, "Speed percentage should be in [0.2, 1]"
        self._speed = round(speed_percent * 255)

    def _write_read_data(self, cmd, read_len):
        with self.serial_lock:
            time.sleep(max(0, self.waiting_gap - (time.time() - self.last_write_time)))
            
            self.ser.write(cmd)
            data = self.ser.read(read_len)
            
            self.last_write_time = time.time()
            return data
    
    def _update_info(self):
        """
        Get the current information about the gripper (width, force, status).
        Refer to: page 66-67, 69-70 of ref [1].
        
            width: real-world width (in meters) of the gripper
            force: the force of the gripper, from 0 to 255.
            status:
                -1: error raised;
                0: opening, completed;
                1: closing, completed;
                2: opening, not completed;
                3: closing, not completed;
        """
        while True:
            data = self._write_read_data(b"\x09\x03\x07\xD0\x00\x03\x04\x0E", 11)
            # Not a valid response
            if data[:3] != b"\x09\x03\x06":
                continue
            width = (255-data[7])/255 * self.max_width
            # Check error flag. Only allow "no fault" and "minor fault: no communication".
            if data[5] != 0x00 and data[5] != 0x09:
                self.info = [width, data[8], -1]
                break
            # Complete Flag.
            if data[3] == 0xF9 or data[3] == 0xB9 or data[3] == 0x79:
                completed = True
            elif data[3] == 0x39:
                completed = False
            else:
                continue
            # Open/close Flag.
            if data[6] == 0xFF:
                g_status = True
            else:
                g_status = False
            status = (1 - int(completed)) * 2 + (int(g_status))
            self.info = [width, data[8], status]
            break

    def _set_force(self, force_percent):
        self._force = round(force_percent * 255)
    
    def _set_width(self, width):
        command = bytearray(b"\x09\x10\x03\xE8\x00\x03\x06\x09\x00\x00\x00\x00\x00")
        command[10] = round(255 - width/self.max_width * 255)
        command[11] = self._speed
        command[12] = self._force
        crc = self._calc_crc(command)
        self._write_read_data(command + crc, 8)

    def _activate(self):
        """
        Activate the gripper.
            Refer to: page 62 of ref [1].
        """
        # Activation Request
        self.ser.write(b"\x09\x10\x03\xE8\x00\x03\x06\x00\x00\x00\x00\x00\x00\x73\x30")
        response = self.ser.read(8)
        if response != b"\x09\x10\x03\xE8\x00\x03\x01\x30":
            raise AssertionError('Unexpected response.')
        time.sleep(self.waiting_gap)
        self.ser.write(b"\x09\x10\x03\xE8\x00\x03\x06\x01\x00\x00\x00\x00\x00\x72\xE1")
        response = self.ser.read(8)
        if response != b"\x09\x10\x03\xE8\x00\x03\x01\x30":
            raise AssertionError('Unexpected response.')
        time.sleep(self.waiting_gap)
        # Read Gripper status until the activation is completed
        self.ser.write(b"\x09\x03\x07\xD0\x00\x01\x85\xCF")
        while self.ser.read(7) != b"\x09\x03\x02\x31\x00\x4C\x15":
            time.sleep(self.waiting_gap)
            self.ser.write(b"\x09\x03\x07\xD0\x00\x01\x85\xCF")

    def _calc_crc(self, command):
        """
        Calculate the Cyclic Redundancy Check (CRC) bytes for command.

        Parameters:
            command: bytes, required, the given command.

        Returns:
            The calculated CRC bytes.
        """
        crc_registor = 0xFFFF
        for data_byte in command:
            tmp = crc_registor ^ data_byte
            for _ in range(8):
                if(tmp & 1 == 1):
                    tmp = tmp >> 1
                    tmp = 0xA001 ^ tmp
                else:
                    tmp = tmp >> 1
            crc_registor = tmp
        crc = bytearray(struct.pack('<H', crc_registor))
        return crc
