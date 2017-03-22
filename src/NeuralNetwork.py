import tensorflow as tf
import numpy as np
import time
import datetime
import os

import Constants as Constant
from DataUtility import Sensor, Gesture, DataSetFormat, DataSetType, File
import DataUtility as DataUtility
import DataHandlers as DataHandlers
import Utility

import NeuralNetworkUtility
from NeuralNetworkUtility import ActivationFunction

# Constants
NEURAL_NETWORK_DATA_PATH = Constant.DATA_SET_FOLDER + "nn_data/"
SESSIONS_FOLDER_NAME = "sessions/"
LOG_FOLDER_NAME = "log/"
EMG_NEURAL_NETWORK_DATA_PATH = NEURAL_NETWORK_DATA_PATH + "emg_network/"

EMG_NEURAL_NETWORK_SESSIONS_FOLDER = EMG_NEURAL_NETWORK_DATA_PATH + SESSIONS_FOLDER_NAME
TRAINING_DATA_FILE_PATH = EMG_NEURAL_NETWORK_DATA_PATH + "training_file.data"

SESS_PATH = None

n_output_nodes = Constant.NUMBER_OF_GESTURES

layer_sizes = [0, 3 * 8, 8, n_output_nodes]  # Network build
layer_activation_functions = [ActivationFunction.SIGMOID, ActivationFunction.SIGMOID, ActivationFunction.SIGMOID]

tf.Session()  # remove warnings... hack...


def set_sess_path(session_name):
    global SESS_PATH
    SESS_PATH = EMG_NEURAL_NETWORK_SESSIONS_FOLDER + "{}/".format(session_name)


def set_default_sess_path():
    set_sess_path(os.listdir(EMG_NEURAL_NETWORK_SESSIONS_FOLDER)[-1])


def select_sess_path():
    session_folder_list = os.listdir(EMG_NEURAL_NETWORK_SESSIONS_FOLDER)

    for i in range(len(session_folder_list)):
        print("{})".format(i), session_folder_list[i])
    session_choice = input("Select a sesssion to use: ")
    try:
        session_choice = int(session_choice)
    except ValueError:
        session_choice = -1

    if session_choice >= len(session_folder_list) or session_choice < 0:
        return

    set_sess_path(session_folder_list[int(session_choice)])


def create_emg_training_file():
    data_handler = DataHandlers.FileDataHandler(DataUtility.TRAINING_FILE_LIST[0])
    n_input_nodes = len(data_handler.get_emg_data_features())
    NeuralNetworkUtility.create_emg_training_file(n_input_nodes, TRAINING_DATA_FILE_PATH, DataUtility.TRAINING_FILE_LIST, Gesture.NUMBER_OF_GESTURES, DataHandlers.FileDataHandler)


def create_emg_network():
    global SESS_PATH
    SESS_PATH = NeuralNetworkUtility.create_emg_network(EMG_NEURAL_NETWORK_SESSIONS_FOLDER, layer_sizes, layer_activation_functions, TRAINING_DATA_FILE_PATH)


def train_emg_network():
    NeuralNetworkUtility.train_emg_network(TRAINING_DATA_FILE_PATH, SESS_PATH)


def test_emg_network():
    summary_list = []

    for test_file in DataUtility.TEST_FILE_LIST:
        data_handler = DataHandlers.FileDataHandler(test_file)

        start_time = time.time()
        results = NeuralNetworkUtility.input_test_emg_network(data_handler, SESS_PATH)
        end_time = time.time()

        recognized_gesture = np.argmax(results)
        print_results(results)

        print("Correct gesture:", Gesture.gesture_to_string(test_file.gesture))
        print("Analyse time:", "{0:.2f}".format(end_time - start_time))

        summary_list.append((test_file.gesture, recognized_gesture))

        print()
        print("File:", test_file.filename)

    print("#############################################################")
    print("Summary List")

    success_list = []
    for i in range(Gesture.NUMBER_OF_GESTURES):
        success_list.append([0, 0])

    for correct_gesture, recognized_gesture in summary_list:

        success_list[correct_gesture][0] += 1

        if correct_gesture == recognized_gesture:
            success_list[correct_gesture][1] += 1

        print(Gesture.gesture_to_string(correct_gesture), " -> ", Gesture.gesture_to_string(recognized_gesture))

    print()
    print("#############################################################")
    print("Success Rate")
    for i in range(Gesture.NUMBER_OF_GESTURES):
        print('{:15s}\t{:4d} of {:4d} -> {:.2f}'.format(Gesture.gesture_to_string(i), success_list[i][1], success_list[i][0], 100 * success_list[i][1] / success_list[i][0]))


def print_results(results):
    for result in results:
        print()
        print("###########################################################")
        for gesture in range(Gesture.NUMBER_OF_GESTURES):
            print('{:15s}\t{:10f}'.format(Gesture.gesture_to_string(gesture), result[gesture]))

    print()
    print("Recognized:", Gesture.gesture_to_string(np.argmax(results)))
