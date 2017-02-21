import json
import os
import numpy

import DataUtility
from DataUtility import DataSetFormat, DataSetType
import Constants as Constant

def get_number_of_arrays_for_sensor(sensor):
    if sensor == DataUtility.Sensor.EMG:
        return Constant.NUMBER_OF_EMG_ARRAYS
    elif sensor == DataUtility.Sensor.ACC:
        return Constant.NUMBER_OF_ACC_ARRAYS
    elif sensor == DataUtility.Sensor.GYR:
        return Constant.NUMBER_OF_GYR_ARRAYS
    elif sensor == DataUtility.Sensor.ORI:
        return Constant.NUMBER_OF_ORI_ARRAYS
    else:
        return None

def get_length_of_arrays_for_sensor(sensor):
    if sensor == DataUtility.Sensor.EMG:
        return Constant.DATA_LENGTH_EMG
    elif sensor == DataUtility.Sensor.ACC:
        return Constant.DATA_LENGTH_ACC
    elif sensor == DataUtility.Sensor.GYR:
        return Constant.DATA_LENGTH_GYR
    elif sensor == DataUtility.Sensor.ORI:
        return Constant.DATA_LENGTH_ORI
    else:
        return None

def get_json_array_name_for_sensor(sensor):
    if sensor == DataUtility.Sensor.EMG:
        return Constant.JSON_EMG_ARRAY_NAME
    elif sensor == DataUtility.Sensor.ACC:
        return Constant.JSON_ACC_ARRAY_NAME
    elif sensor == DataUtility.Sensor.GYR:
        return Constant.JSON_GYR_ARRAY_NAME
    elif sensor == DataUtility.Sensor.ORI:
        return Constant.JSON_ORI_ARRAY_NAME
    else:
        return None

# Function: get_json_data_from_file
# ----------------------------
#   Open JSON-file
#
# 	file : JSON-file to open
#
#   returns : JSON-data from file
#
def get_json_data_from_file(file):
    with open(file.get_file_path()) as json_file:
        json_data = json.load(json_file)

    return json_data

# Function: is_file_already_compressed
# ----------------------------
#   Check if file already are compressed
#
# 	file : JSON-file to compress
# 	data_set_type : Training or test data set
#
#   returns : true if file exist in compressed folder, false else
#
def is_file_already_compressed(file, data_set_type):
    compressed_file_path = DataUtility.get_data_set_path(DataSetFormat.COMPRESSED, data_set_type) + file.filename
    return os.path.exists(compressed_file_path)

# Function: compress_json_file
# ----------------------------
#   compress input json file
#
# 	file : JSON-file to compress
# 	data_set_type : Training or test data set
#
def compress_json_file(file, data_set_type):
    print("Compressing file: " + file.filename)
    raw_data = get_json_data_from_file(file)

    compressed_data = {}

    json_array_name_list = [Constant.JSON_EMG_ARRAY_NAME, Constant.JSON_ACC_ARRAY_NAME, Constant.JSON_GYR_ARRAY_NAME, Constant.JSON_ORI_ARRAY_NAME]
    data_length_list = [Constant.DATA_LENGTH_EMG, Constant.DATA_LENGTH_ACC, Constant.DATA_LENGTH_GYR, Constant.DATA_LENGTH_ORI]

    for json_array_name , data_length in zip(json_array_name_list, data_length_list):
        compressed_data[json_array_name] = {}
        transposed_raw_data = numpy.transpose(raw_data[json_array_name][Constant.JSON_ARRAY_DATA_TABLE_NAME][:data_length]).tolist()
        compressed_data[json_array_name][Constant.JSON_ARRAY_DATA_TABLE_NAME] = transposed_raw_data
        
    compressed_file_path = DataUtility.get_data_set_path(DataSetFormat.COMPRESSED, data_set_type) + file.filename
    with open(compressed_file_path, 'w') as outfile:
        json.dump(compressed_data, outfile)
