import os

import Constants as Constant



class Sensor:
    NUMBER_OF_SENSORS = Constant.NUMBER_OF_SENSORS

    EMG = 0
    ACC = 1
    ORI = 2
    GYR = 3
    EMPTY_SENSOR = NUMBER_OF_SENSORS

    SENSOR_NAMES = [ "Emg", "Acc", "Ori", "Gyr", "Empty sensor" ]

    @staticmethod
    def sensor_to_string(sensor):
        if sensor >= Sensor.NUMBER_OF_SENSORS:
            return Sensor.SENSOR_NAMES[Sensor.NUMBER_OF_SENSORS]
        else:
            return Sensor.SENSOR_NAMES[sensor]

class Gesture:
    NUMBER_OF_GESTURES = Constant.NUMBER_OF_GESTURES

    EAT = 0
    HELP = 1
    SLEEP = 2
    THANKYOU = 3
    WHY = 4
    NO = 5
    YES = 6
    DRINK = 7
    HELLO = 8
    SORRY = 9

    NONE_GESTURE = NUMBER_OF_GESTURES

    GESTURE_NAMES = [ "EAT", "HELP", "SLEEP", "THANKYOU", "WHY", "NO", "YES", "DRINK", "HELLO", "SORRY", "NONE_GESTURE" ]

    @staticmethod
    def gesture_to_string(gesture):
        if gesture >= Gesture.NUMBER_OF_GESTURES:
            return Gesture.GESTURE_NAMES[Gesture.NUMBER_OF_GESTURE]
        else:
            return Gesture.GESTURE_NAMES[gesture]

class DataSetFormat:
    COMPRESSED = 0
    RAW = 1

class DataSetType:
    TRAINING = 0
    TEST = 1
    RECORDED = 2

class File:
    def __init__(self, filename, gesture):
        self.filename = filename
        self.gesture = gesture
        self.path = ""
        self.is_recorded = False

    def __init__(self, path, filename, gesture, is_recorded=False):
        self.filename = filename
        self.gesture = gesture
        self.path = path
        self.is_recorded = is_recorded

    def get_file_path(self):
        return self.path + self.filename

# Function: get_data_set_path
# ----------------------------
#   returns the data set path to the parameter path
#
# 	format : Compressed or raw data set
# 	type : Training or test data set
#
#   returns : the path to the data set
#
def get_data_set_path(data_format, data_type):
    path = "../" + Constant.DATA_SET_FOLDER

    format_folder = Constant.COMPRESSED_DATA_FOLDER
    if (data_format == DataSetFormat.RAW):
        format_folder = Constant.RAW_DATA_FOLDER
    path = path + format_folder

    type_folder = Constant.TEST_SET_FOLDER
    if (data_type == DataSetType.TRAINING):
        type_folder = Constant.TRAINING_SET_FOLDER
    elif data_type == DataSetType.RECORDED:
        type_folder = Constant.RECORDED_SET_FOLDER
    path = path + type_folder

    return path

def get_gesture_from_filename(filename):
    gesture = 0
    for gesture_name in Gesture.GESTURE_NAMES[:-1]:
        if gesture_name.lower() in filename.lower():
            return gesture

        gesture = gesture + 1

def generate_file_list(data_folder_path):
    filelist = []
    for filename in os.listdir(data_folder_path):
        gesture = get_gesture_from_filename(filename)
        if "recorded" in filename.lower():
            filelist.append(File(data_folder_path, filename, gesture))
        else:
            filelist.append(File(data_folder_path, filename, gesture, is_recorded=True))
    return filelist

def get_gesture_file_count_in_folder(folder_path):
    file_list = generate_file_list(folder_path)

    gesture_file_counts = [0] * Gesture.NUMBER_OF_GESTURES

    for f in file_list:
        gesture_file_counts[f.gesture] += 1

    return gesture_file_counts


TRAINING_FILE_LIST = generate_file_list(get_data_set_path(DataSetFormat.COMPRESSED, DataSetType.TRAINING))
#TEST_FILE_LIST = generate_file_list(get_data_set_path(DataSetFormat.COMPRESSED, DataSetType.TEST))
TEST_FILE_LIST = generate_file_list(get_data_set_path(DataSetFormat.COMPRESSED, DataSetType.RECORDED))
TRAINING_FILE_LIST.extend(generate_file_list(get_data_set_path(DataSetFormat.COMPRESSED, DataSetType.RECORDED)))
#TRAINING_FILE_LIST.extend(TEST_FILE_LIST)
