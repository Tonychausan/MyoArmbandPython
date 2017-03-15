import myo as libmyo
import os
import time

import DataHandlers
import Constants as Constant
import Utility

from DataUtility import Sensor, Gesture, DataSetFormat, DataSetType, File

start_time = 0.0


class LiveMessureListener(libmyo.DeviceListener):
    def __init__(self):
        self.emg = None
        self.acc = None
        self.gyr = None
        self.ori = None

    def on_pair(self, myo, timestamp, firmware_version):
        print("Hello, Myo!")
        myo.set_stream_emg(libmyo.StreamEmg.enabled)

    def on_unpair(self, myo, timestamp):
        print("Goodbye, Myo!")

    def on_arm_sync(self, myo, *args):
        # myo.set_stream_emg(libmyo.StreamEmg.enabled)
        pass

    def on_orientation_data(self, myo, timestamp, quat):
        # print("Orientation:", quat.x, quat.y, quat.z, quat.w)
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


######################################################################
#
#
######################################################################

class LiveGestureListener(libmyo.DeviceListener):
    def __init__(self):
        self.data_handler = DataHandlers.InputDataHandler()
        self.is_recording = False

        self.is_sensor_recording = []
        for sensor in range(Sensor.NUMBER_OF_SENSORS):
            self.is_sensor_recording.append(False)

        self.sensor_array_length_list = []
        for sensor in range(Sensor.NUMBER_OF_SENSORS):
            self.sensor_array_length_list.append(Utility.get_length_of_arrays_for_sensor(sensor))

    def reset_stats(self):
        self.data_handler.reset_data()

        for sensor in range(Sensor.NUMBER_OF_SENSORS):
            self.is_sensor_recording[sensor] = True
        self.is_recording = True

    def expand_data_length(self, time_margin):
        for sensor in range(Sensor.NUMBER_OF_SENSORS):
            self.sensor_array_length_list[sensor] += time_margin * Utility.get_frequency_of_sensor(sensor)

    def on_pair(self, myo, timestamp, firmware_version):
        print("Hello, Myo!")
        myo.set_stream_emg(libmyo.StreamEmg.enabled)

    def on_unpair(self, myo, timestamp):
        print("Goodbye, Myo!")

    def on_arm_sync(self, myo, *args):
        # myo.set_stream_emg(libmyo.StreamEmg.enabled)
        pass

    def on_orientation_data(self, myo, timestamp, quat):
        # print("Orientation:", quat.x, quat.y, quat.z, quat.w)
        sensor = Sensor.ORI
        if self.is_sensor_recording[sensor]:
            self.data_handler.append_data(sensor, quat)
            self.have_sensor_finished(sensor)

    def on_accelerometor_data(self, myo, timestamp, acceleration):
        sensor = Sensor.ACC
        if self.is_sensor_recording[sensor]:
            self.data_handler.append_data(sensor, acceleration)
            self.have_sensor_finished(sensor)

    def on_gyroscope_data(self, myo, timestamp, gyroscope):
        sensor = Sensor.GYR
        if self.is_sensor_recording[sensor]:
            self.data_handler.append_data(sensor, gyroscope)
            self.have_sensor_finished(sensor)

    def on_emg_data(self, myo, timestamp, emg):
        sensor = Sensor.EMG
        if self.is_sensor_recording[sensor]:
            self.data_handler.append_data(sensor, emg)
            self.have_sensor_finished(sensor)

        if self.is_finished() and self.is_recording:
            self.recording_off()

    def have_sensor_finished(self, sensor):
        current_data_length = self.data_handler.get_data_length(sensor)
        if current_data_length >= self.sensor_array_length_list[sensor]:
            self.is_sensor_recording[sensor] = False

    def is_finished(self):
        number_of_finished_sensors = 0
        for sensor in range(Sensor.NUMBER_OF_SENSORS):
            if not self.is_sensor_recording[sensor]:
                number_of_finished_sensors += 1

        if number_of_finished_sensors >= Sensor.NUMBER_OF_SENSORS:
            return True

        return False

    def recording_on(self):
        global start_time
        print("Recoring is on...")

        self.reset_stats()
        self.is_recording = True

        start_time = time.time()

    def recording_off(self):
        self.is_recording = False
        print("Recoring finished!")
        print("--- %s seconds ---" % (time.time() - start_time))
