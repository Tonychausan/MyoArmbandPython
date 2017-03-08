import myo as libmyo
import os
import json
import numpy
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
    menu_item_list.append(MenuItem("Pre-data gesture test", compare_prerecorded_files))
    menu_item_list.append(MenuItem("Create Gesture-files", create_gesture_files))
    menu_item_list.append(MenuItem("Neural Network Menu", neural_network_testing))
    return menu_item_list

def print_menu(menu_item_list):
    os.system('cls')
    print("Main Menu")
    print("###################################################")
    for i in range(len(menu_item_list)):
        print(str(i) + ")", menu_item_list[i].menu_text)
    print()

    action = check_int_input_value(min=0, max=len(menu_item_list))
    os.system('cls')
    return action

def check_int_input_value(min, max, empty_input_allowed=False):
    action = min - 1
    while action < min:
        action = input( "Choose an action: " )
        if empty_input_allowed and action == "":
            return min-1
        try:
            action = int(action)
        except ValueError:
            print("That's not an int!")
            action = -1
            continue

        if action >= max:
            print ("That's a high int!")
            action = min-1
        elif action < 0:
            print ("That's a low int!")
            action = min-1

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
    nn_menu_list.append(MenuItem("Train network", NeuralNetwork.continue_emg_network_training))
    nn_menu_list.append(MenuItem("Test data", NeuralNetwork.test_emg_network))

    while True:
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
            results = NeuralNetwork.input_test_emg_network(listener.data_handler)
            NeuralNetwork.print_results(results)


    except KeyboardInterrupt:
        print('\nQuit')


    hub.shutdown()  # !! crucial

def create_gesture_files():
    libmyo.init('../myo-sdk-win-0.9.0/bin')
    listener = DeviceListener.LiveGestureListener()
    listener.expand_data_length(time_margin=1.5)
    hub = libmyo.Hub()
    hub.run(2000, listener)

    print("Create record data set")

    last_file = None;
    try:
        while True:
            print()
            print("#################################################################################\n",end="")
            input("Press Enter to continue...")
            listener.recording_on()
            sleep(2.0)
            while listener.is_recording:
                pass

            folder_path = DataUtility.get_data_set_path(DataSetFormat.RAW, DataSetType.RECORDED)

            results = NeuralNetwork.input_test_emg_network(listener.data_handler)
            recognized_gesture = numpy.argmax(results)
            NeuralNetwork.print_results(results)

            # Print number to gesture table
            print()
            for gesture in range(Gesture.NUMBER_OF_GESTURES):
                print(gesture, "->", Gesture.gesture_to_string(gesture), "         ", end="")
            print()
            print(Gesture.NUMBER_OF_GESTURES, "->", "remove last file...")
            print(Gesture.NUMBER_OF_GESTURES + 1, "->", "continue...")

            gesture_recorded = check_int_input_value(min=0, max=Gesture.NUMBER_OF_GESTURES + 2,  empty_input_allowed=True)
            if gesture_recorded == -1:
                gesture_recorded = recognized_gesture

            if(gesture_recorded == Gesture.NUMBER_OF_GESTURES):
                if(last_file != None):
                    os.remove(last_file.get_file_path())
                    print("Removed file:", last_file.filename)
                    last_file = None
                else:
                    print("No last file, remove it manually")
                continue
            if(gesture_recorded == Gesture.NUMBER_OF_GESTURES + 1):
                continue

            gesture_file_count_list = DataUtility.get_gesture_file_count_in_folder(folder_path)
            file_number = gesture_file_count_list[gesture_recorded]
            filename = "recorded-" + Gesture.gesture_to_string(gesture_recorded) + "-" + str(file_number) + ".json"

            last_file = listener.data_handler.create_json_file(filename)


    except KeyboardInterrupt:
        print('\nQuit')


    hub.shutdown()  # !! crucial
