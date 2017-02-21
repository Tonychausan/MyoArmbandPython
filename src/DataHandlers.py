import json
import numpy

import DataUtility as DataUtility
import Constants as Constant
import Utility as Utility
from DataUtility import Sensor, Gesture, DataSetFormat, DataSetType

class DataHandler:
    def __init__(self):
        self.emg_data = []
        self.acc_data = []
        self.gyr_data = []
        self.ori_data = []

    def set_sensor_data(self, data, sensor):
        if sensor == Sensor.EMG:
            self.emg_data = data
        elif sensor == Sensor.ACC:
            self.acc_data = data
        elif sensor == Sensor.GYR:
            self.gyr_data = data
        elif sensor == Sensor.ORI:
            self.ori_data = data
        else:
            pass

    def get_sensor_data(self, sensor):
        if sensor == Sensor.EMG:
            return self.emg_data
        elif sensor == Sensor.ACC:
            return self.acc_data
        elif sensor == Sensor.GYR:
            return self.gyr_data
        elif sensor == Sensor.ORI:
            return self.ori_data
        else:
            return []

class FileDataHandler(DataHandler):
    def __init__(self, file):
        super().__init__()
        self.file = file
        self.file_to_data()

    def json_data_to_data_by_sensor(self, json_data, sensor):
        json_array_name = Utility.get_json_array_name_for_sensor(sensor)
        json_data_table_name = Constant.JSON_ARRAY_DATA_TABLE_NAME

        data_array = json_data[json_array_name][json_data_table_name]


        self.set_sensor_data(data_array, sensor)

    # Function: file_to_data
    # ----------------------------
    #   Generate data from the json file
    #
    def file_to_data(self):
        with open(self.file.get_file_path()) as json_file:
            json_data = json.load(json_file)

        for sensor in range(0, Sensor.NUMBER_OF_SENSORS):
            self.json_data_to_data_by_sensor(json_data, sensor)
