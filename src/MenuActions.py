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
import DeviceListener

class MenuItem:
    def __init__(self, menu_text, function):
        self.menu_text = menu_text
        self.function = function


def create_main_menu():
    menu_item_list = []
    menu_item_list.append(MenuItem("Try gestures", live_gesture_recognition))
    menu_item_list.append(MenuItem("Measurment display", print_myo_data))
    menu_item_list.append(MenuItem("Compress Files", compress_json_files))
    menu_item_list.append(MenuItem("Delete all compressed files", remove_all_compressed_files))
    menu_item_list.append(MenuItem("Pre-data gesture comparison", compare_prerecorded_files))
    menu_item_list.append(MenuItem("Create gesture files", create_gesture_files))
    menu_item_list.append(MenuItem("Test Neural Network", neural_network_testing))
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
        action = check_int_input_value(action, min=0, max=len(menu_item_list))
    return action

def check_int_input_value(action, min, max):
    try:
        action = int(action)
    except ValueError:
        print("That's not an int!")
        action = -1
        return action

    if action >= max:
        print ("That's a high int!")
        action = -1
    elif action < 0:
        print ("That's a low int!")
        action = -1

    return action

def print_myo_data():
    libmyo.init('../myo-sdk-win-0.9.0/bin')
    listener = DeviceListener.LiveMessureListener()
    hub = libmyo.Hub()
    hub.run(200, listener)

    try:
        while True:
            sleep(0.5)
            listener.print_data()
    except KeyboardInterrupt:
        print('\nQuit')

    hub.shutdown()  # !! crucial


def compress_json_files():
    print("Compressing JSON-files")

    for data_set_type in [DataSetType.TRAINING, DataSetType.TEST, DataSetType.RECORDED]:
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

def live_gesture_recognition():
    print("Try Gesture")
    libmyo.init('../myo-sdk-win-0.9.0/bin')
    listener = DeviceListener.LiveGestureListener()
    hub = libmyo.Hub()
    hub.run(2000, listener)

    try:
        while True:
            print()
            input("Press Enter to continue...")
            listener.recording_on()
            while listener.is_recording:
                pass
            NeuralNetwork.input_test_emg_network(listener.data_handler)


    except KeyboardInterrupt:
        print('\nQuit')


    hub.shutdown()  # !! crucial

def create_gesture_files():
    folder_path = '../data/raw_files/recorded_set/'

    print("Try Gesture")
    libmyo.init('../myo-sdk-win-0.9.0/bin')
    listener = DeviceListener.LiveGestureListener()
    listener.expand_data_length(time_margin=1.5)
    hub = libmyo.Hub()
    hub.run(2000, listener)

    try:
        while True:
            print()
            input("Press Enter to continue...")
            listener.recording_on()
            sleep(2.0)
            while listener.is_recording:
                pass

            folder_path = DataUtility.get_data_set_path(DataSetFormat.RAW, DataSetType.RECORDED)
            gesture_file_count_list = DataUtility.get_gesture_file_count_in_folder(folder_path)

            # Print number to gesture table
            print()
            for gesture in range(Gesture.NUMBER_OF_GESTURES):
                print(gesture, "->", Gesture.gesture_to_string(gesture), "\t" , end="")
            print(Gesture.NUMBER_OF_GESTURES, "->", "exit...")

            gesture_recorded = -1
            while gesture_recorded == -1:
                gesture_recorded = input("Recorded gesture: ")
                gesture_recorded = check_int_input_value(gesture_recorded , min=0, max=Gesture.NUMBER_OF_GESTURES + 1)

            if(gesture_recorded == Gesture.NUMBER_OF_GESTURES):
                continue

            file_number = gesture_file_count_list[gesture_recorded]
            filename = "recorded-" + Gesture.gesture_to_string(gesture_recorded) + "-" + str(file_number)

            listener.data_handler.create_json_file(filename)



    except KeyboardInterrupt:
        print('\nQuit')


    hub.shutdown()  # !! crucial
