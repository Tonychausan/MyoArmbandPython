import myo as libmyo
import os

class Listener(libmyo.DeviceListener):
    def __init__(self):
        self.emg = None
        self.acc = None
        self.gyr = None
        self.ori = None

    def on_pair(self, myo, timestamp, firmware_version):
        print("Hello, Myo!")

    def on_unpair(self, myo, timestamp):
        print("Goodbye, Myo!")

    def on_arm_sync(self, myo, *args):
        myo.set_stream_emg(libmyo.StreamEmg.enabled)

    def on_orientation_data(self, myo, timestamp, quat):
        #print("Orientation:", quat.x, quat.y, quat.z, quat.w)
        self.ori = quat
        pass

    def on_accelerometor_data(self, myo, timestamp, acceleration):
        self.acc = acceleration
        pass

    def on_gyroscope_data(self, myo, timestamp, gyroscope):
        self.gyr = gyroscope
        pass

    def on_emg_data(self, myo, timestamp, emg):
        self.emg = emg
        pass

    def print_data(self):
        os.system('cls')
        print("EMG: ", self.emg)
        print("Accelerometer: ", self.acc)
        print("Gyroscope: ", self.gyr)
        print("Orientation: ", self.ori)
