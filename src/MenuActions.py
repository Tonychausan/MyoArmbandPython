import myo as libmyo
import os
import json
from time import sleep

import DataUtility as DataUtility
from DataUtility import DataSetFormat, DataSetType

import Constants as Constant
import Utility as Utility

def print_myo_data(listener):
    try:
        while True:
            sleep(0.5)
            listener.print_data();
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
