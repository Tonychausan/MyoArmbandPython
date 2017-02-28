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
import NeuralNetwork

class MenuItem:
    def __init__(self, menu_text, function):
        self.menu_text = menu_text
        self.function = function

class MainMenuItem(MenuItem):
    def __init__(self, menu_text, function, is_myo_depended):
        super().__init__(menu_text, function)
        self.is_myo_depended = is_myo_depended

def create_main_menu():
    menu_item_list = []
    menu_item_list.append(MainMenuItem("Try gestures", live_gesture_recognition, True))
    menu_item_list.append(MainMenuItem("Measurment display", print_myo_data, True))
    menu_item_list.append(MainMenuItem("Compress Files", compress_json_files, False))
    menu_item_list.append(MainMenuItem("Delete all compressed files", remove_all_compressed_files, False))
    menu_item_list.append(MainMenuItem("Pre-data gesture comparison", compare_prerecorded_files, False))
    menu_item_list.append(MainMenuItem("Test Neural Network", neural_network_testing, False))
    return menu_item_list

def print_menu(menu_item_list):
    os.system('cls')
    print("Main Menu")
    print("###################################################")
    for i in range(len(menu_item_list)):
        print(str(i) + ")", menu_item_list[i].menu_text)
    print()
    action = -1
    while action == -1:
        action = input( "Choose an action: " )
        action = check_menu_action(action, menu_item_list)
    return action

def check_menu_action(action, menu_item_list):
    try:
        action = int(action)
    except ValueError:
        print("That's not an int!")
        action = -1
        return action

    if action >= len(menu_item_list):
        print ("That's a high int!")
        action = -1
    elif action < 0:
        print ("That's a low int!")
        action = -1

    return action

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


def neural_network_testing():
    nn_menu_list = []
    nn_menu_list.append(MenuItem("Create training file", NeuralNetwork.create_emg_training_file))
    nn_menu_list.append(MenuItem("Create network", NeuralNetwork.create_emg_network))
    nn_menu_list.append(MenuItem("Train network", NeuralNetwork.continue_emg_training))
    nn_menu_list.append(MenuItem("Test data", NeuralNetwork.test_emg_network))

    os.system('cls')
    print("Neural Network menu")
    print("####################################################")
    action = print_menu(nn_menu_list)
    nn_menu_list[action].function()

    input("Press Enter to continue...")

def live_gesture_recognition(listener):
    print("Try Gesture")
