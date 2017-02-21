import myo as libmyo
import os
import json
from time import sleep

import DataUtility as DataUtility
from DataUtility import DataSetFormat, DataSetType, Gesture, Sensor

import Constants as Constant
import Utility as Utility
import DataHandlers
import CompareMethods

def print_myo_data(listener):
    try:
        while True:
            sleep(0.5)
            listener.print_data()
    except KeyboardInterrupt:
        print('\nQuit')


def compress_json_files():
    print("Compressing JSON-files")

    for data_set_type in [DataSetType.TRAINING, DataSetType.TEST]:
        path = DataUtility.get_data_set_path(DataSetFormat.RAW, data_set_type)
        raw_filelist = DataUtility.generate_file_list(path)

        for file in raw_filelist:
            if Utility.is_file_already_compressed(file, data_set_type):
                continue

            Utility.compress_json_file(file, data_set_type)

    print("Finshed compressing!")

def compare_prerecorded_files():
    print("Compare pre-recorded tests")
    for file in DataUtility.TEST_FILE_LIST:
        print("#####################################################################################")
        print("File: " + file.filename)
        data_handler = DataHandlers.FileDataHandler(file)
        recognized_gesture = CompareMethods.cross_correlation_comparison(data_handler)

        print("Gesture: " + Gesture.gesture_to_string(file.gesture))
        print("Regcognized Gesture: " + Gesture.gesture_to_string(recognized_gesture))
        print("\n\n\n")


def remove_all_compressed_files():
    print("Removing compressed files...")
    for file in DataUtility.TEST_FILE_LIST:
        print(file.get_file_path())
        os.remove(file.get_file_path())

    for file in DataUtility.TRAINING_FILE_LIST:
        print(file.get_file_path())
        os.remove(file.get_file_path())

    print("Finished")

def live_gesture_recognition():
    pass
