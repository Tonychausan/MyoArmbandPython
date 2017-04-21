import myo as libmyo
import os
import time
import numpy
from time import sleep

import DataUtility as DataUtility
from DataUtility import DataSetFormat, DataSetType, Gesture, Sensor
from MenuUtility import MenuItem, print_menu, is_valid_menu_item

import Constants as Constant
import Utility as Utility
import DataHandlers
import CompareMethods
import NeuralNetwork
import DeviceListener
import HackathonDataNeuralNetwork
import NeuralNetworkUtility


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
    DataFile = HackathonDataNeuralNetwork
    is_hackathon = True

    network_session = NeuralNetworkUtility.NeuralNetwork(DataFile.SESSION_FOLDERS, DataFile.DATA_HANDLER_TYPE, is_hackathon)
    nn_menu_list = []
    nn_menu_list.append(MenuItem("Change dataset", network_session.change_dataset))
    nn_menu_list.append(MenuItem("Select a session", network_session.select_sess_path))
    nn_menu_list.append(MenuItem("Create network", network_session.create_emg_network))
    nn_menu_list.append(MenuItem("Train network", network_session.train_emg_network))
    nn_menu_list.append(MenuItem("Test data", network_session.test_emg_network))
    nn_menu_list.append(MenuItem("Result analyses", network_session.result_analyses))

    while True:
        os.system('cls')
        print("Neural Network menu")
        print("current session path:", network_session.sess_path)

        print()
        network_session.print_sess_info(network_session.sess_path)
        print()
        print("####################################################")
        action = print_menu(nn_menu_list)
        nn_menu_list[action].function()


def live_gesture_recognition():
    print("Try Gesture")
    libmyo.init('../myo-sdk-win-0.9.0/bin')
    listener = DeviceListener.LiveGestureListener()
    hub = libmyo.Hub()
    hub.run(2000, listener)

    try:
        while True:
            print("#################################################################################\n", end="")
            print()
            try:
                input("press enter to continue... ")
            except EOFError:
                time.sleep(1)

            listener.recording_on()
            while listener.is_recording:
                pass

            print()
            network_session = NeuralNetworkUtility.NeuralNetwork(NeuralNetwork.SESSION_FOLDERS, NeuralNetwork.DATA_HANDLER_TYPE, False)
            network_session.set_sess_path("2017-04-20-1454")
            network_session.get_network_meta_data()

            results = network_session.input_test_emg_network(listener.data_handler)
            network_session.print_results(results)

            recognized_gesture = numpy.argmax(results)
            print()
            print("Recognized:", Gesture.gesture_to_string(recognized_gesture))

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

    last_file = None
    try:
        while True:
            print()
            print("#################################################################################\n", end="")
            try:
                a = input("Enter 'r' to remove last file, or press enter to continue... ")
            except EOFError:
                time.sleep(1)
            if a == "r":
                gesture_recorded = Gesture.NUMBER_OF_GESTURES
            else:
                listener.recording_on()
                sleep(2.0)
                while listener.is_recording:
                    pass

                folder_path = DataUtility.get_data_set_path(DataSetFormat.RAW, DataSetType.RECORDED)

                network_session = NeuralNetworkUtility.NeuralNetwork(NeuralNetwork.SESSION_FOLDERS, NeuralNetwork.DATA_HANDLER_TYPE, False)
                network_session.set_sess_path("2017-04-20-1454")
                network_session.get_network_meta_data()

                results = network_session.input_test_emg_network(listener.data_handler)
                recognized_gesture = numpy.argmax(results)

                print()
                print("###########################################################")
                print()
                network_session.print_results(results)

                # Print number to gesture table
                print()
                print("{})".format(Gesture.NUMBER_OF_GESTURES), "remove last file...")
                print("{})".format(Gesture.NUMBER_OF_GESTURES + 1), "continue...")

                print()
                print("Recognized:", Gesture.gesture_to_string(recognized_gesture))

                gesture_recorded = -1
                while gesture_recorded < 0 or gesture_recorded >= Gesture.NUMBER_OF_GESTURES + 2:
                    gesture_recorded = input("Correction: ")
                    if gesture_recorded == "":
                        gesture_recorded = recognized_gesture
                    elif not Utility.is_int_input(gesture_recorded):
                        gesture_recorded = -1
                    else:
                        gesture_recorded = int(gesture_recorded)

            if(gesture_recorded == Gesture.NUMBER_OF_GESTURES):
                if last_file is not None:
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
