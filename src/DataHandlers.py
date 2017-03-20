import json
import numpy
import pywt

import DataUtility as DataUtility
import Constants as Constant
import Utility as Utility
from DataUtility import Sensor, DataSetFormat, DataSetType

EMG_WAVELET_LEVEL = 1
NUMBER_OF_FEATURES = 3

feature_functions = [Utility.mean_absolute_value, Utility.root_mean_square, Utility.waveform_length]


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

    def get_emg_sums_normalized2(self):
        emg_sums = []
        emg_sum_min = -1
        emg_sum_max = -1
        for emg_array in self.emg_data:
            emg_sum = numpy.sum(numpy.square(emg_array[:Constant.DATA_LENGTH_EMG]))
            emg_sums.append(emg_sum)

            if emg_sum_min == -1:
                emg_sum_min = emg_sum
                emg_sum_max = emg_sum
            elif emg_sum < emg_sum_min:
                emg_sum_min = emg_sum
            elif emg_sum > emg_sum_max:
                emg_sum_max = emg_sum

        emg_sums = Utility.NormalizeArray(emg_sums)
        emg_sums = numpy.append(emg_sums, self.get_waveform_length_of_emg()).flatten()

        return emg_sums

    def get_waveform_length_of_emg(self):
        emg_waveform_length_list = []

        for emg_array in self.emg_data:
            emg_waveform = numpy.subtract(emg_array[:Constant.DATA_LENGTH_EMG - 1], emg_array[1:Constant.DATA_LENGTH_EMG])
            emg_waveform = numpy.sum(numpy.absolute(emg_waveform))
            emg_waveform_length_list.append(emg_waveform)

        emg_waveform_length_list = Utility.NormalizeArray(emg_waveform_length_list)
        return emg_waveform_length_list

    # http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7150944&tag=1
    def wavelet_feature_extraxtion(self):
        emg_feature_data = []

        n = EMG_WAVELET_LEVEL
        for emg_id in range(len(self.emg_data)):
            emg_array = self.emg_data[emg_id]

            coefficient_subsets = pywt.wavedec(emg_array, 'db1', level=n)
            cAn = coefficient_subsets[0]  # approximation coefficient
            cD = coefficient_subsets[1:]  # detail coefficient subset

            reconstructed_signals = []
            for i in range(n + 1):
                temp_coeffs = [None] * (n + 1 - i)  # init list [cAn, cD(n-i), cD(n-i-1)..., cD1]
                if i == 0:
                    temp_coeffs[0] = cAn
                else:
                    temp_coeffs.append(None)
                    temp_coeffs[1] = cD[i - 1]

                reconstructed_signals.append(pywt.waverec(temp_coeffs, 'db1'))

            # Emg signal = An + Dn + D(n-1) + ... + D1
            # An = reconstructed_signals[0]
            # D = reconstructed_signals[1:]

            number_of_coeffcient_subsets = len(coefficient_subsets)
            number_of_reconstructed_signals = len(reconstructed_signals)

            for feature_function_id in range(len(feature_functions)):
                feature_function = feature_functions[feature_function_id]

                if emg_id == 0:
                    for i in range(number_of_coeffcient_subsets):
                        emg_feature_data.append([feature_function(coefficient_subsets[i])])

                    for i in range(number_of_reconstructed_signals):
                        emg_feature_data.append([feature_function(reconstructed_signals[i])])
                else:
                    emg_feature_data_id = feature_function_id * (number_of_coeffcient_subsets + number_of_reconstructed_signals)
                    for i in range(number_of_coeffcient_subsets):
                        j = emg_feature_data_id + i
                        emg_feature_data[j].append(feature_function(coefficient_subsets[i]))

                    for i in range(number_of_reconstructed_signals):
                        j = emg_feature_data_id + number_of_coeffcient_subsets + i
                        emg_feature_data[j].append(feature_function(reconstructed_signals[i]))

        for i in range(len(emg_feature_data)):
            emg_feature_data[i] = Utility.NormalizeArray(emg_feature_data[i])

        return numpy.array(emg_feature_data).flatten()

    def raw_emg_feature_extraxtion(self):
        emg_feature_data = []
        for emg_id in range(len(self.emg_data)):
            emg_array = self.emg_data[emg_id]
            for feature_function_id in range(len(feature_functions)):
                feature_function = feature_functions[feature_function_id]
                if emg_id == 0:
                    emg_feature_data.append([feature_function(emg_array)])
                else:
                    emg_feature_data[feature_function_id].append(feature_function(emg_array))

        for i in range(len(emg_feature_data)):
            emg_feature_data[i] = Utility.NormalizeArray(emg_feature_data[i])

        return emg_feature_data

    def get_emg_data_features(self):
        emg_data_features = numpy.append(self.wavelet_feature_extraxtion(), self.raw_emg_feature_extraxtion())
        return emg_data_features


class InputDataHandler(DataHandler):
    def __init__(self):
        super().__init__()
        self.reset_data()

    def init_data_length_variables(self):
        self.emg_data_length = 0
        self.acc_data_length = 0
        self.gyr_data_length = 0
        self.ori_data_length = 0

    def increment_data_length_variable(self, sensor):
        if sensor == Sensor.EMG:
            self.emg_data_length += 1
        elif sensor == Sensor.ACC:
            self.acc_data_length += 1
        elif sensor == Sensor.GYR:
            self.gyr_data_length += 1
        elif sensor == Sensor.ORI:
            self.ori_data_length += 1
        else:
            return None

    def init_empyt_data(self):
        for i in range(Constant.NUMBER_OF_EMG_ARRAYS):
            self.emg_data.append([])
        for i in range(Constant.NUMBER_OF_ACC_ARRAYS):
            self.acc_data.append([])
        for i in range(Constant.NUMBER_OF_GYR_ARRAYS):
            self.gyr_data.append([])
        for i in range(Constant.NUMBER_OF_ORI_ARRAYS):
            self.ori_data.append([])

    def append_data(self, sensor, data):
        if sensor == Sensor.EMG:
            sensor_data = self.emg_data
        elif sensor == Sensor.ACC:
            sensor_data = self.acc_data
        elif sensor == Sensor.GYR:
            sensor_data = self.gyr_data
        elif sensor == Sensor.ORI:
            data = [data.x, data.y, data.z, data.w]
            sensor_data = self.ori_data
        else:
            return None

        for i in range(Utility.get_number_of_arrays_for_sensor(sensor)):
            sensor_data[i].append(data[i])

        self.increment_data_length_variable(sensor)

    def reset_data(self):
        self.emg_data = []
        self.acc_data = []
        self.gyr_data = []
        self.ori_data = []
        self.init_empyt_data()
        self.init_data_length_variables()

    def get_data_length(self, sensor):
        if sensor == Sensor.EMG:
            return self.emg_data_length
        elif sensor == Sensor.ACC:
            return self.acc_data_length
        elif sensor == Sensor.GYR:
            return self.gyr_data_length
        elif sensor == Sensor.ORI:
            return self.ori_data_length
        else:
            return None

    def create_json_file(self, filename):
        print("Creating file:", filename)

        json_data = {}
        for sensor in range(Sensor.NUMBER_OF_SENSORS):
            json_array_name = Utility.get_json_array_name_for_sensor(sensor)
            json_data_table_name = Constant.JSON_ARRAY_DATA_TABLE_NAME

            json_data[json_array_name] = {}
            json_data[json_array_name][json_data_table_name] = self.get_sensor_data(sensor)

        folder_path = DataUtility.get_data_set_path(DataSetFormat.RAW, DataSetType.RECORDED)
        with open(folder_path + filename, 'w') as outfile:
            json.dump(json_data, outfile)

        o_file = DataUtility.File(folder_path, filename, None)

        return o_file


class FileDataHandler(DataHandler):
    def __init__(self, file_data):
        super().__init__()
        self.file = file_data
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
