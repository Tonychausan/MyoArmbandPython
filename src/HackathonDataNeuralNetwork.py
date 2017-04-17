import numpy
import os
import time
import tensorflow as tf
import re

import DataHandlers
import NeuralNetworkUtility
from NeuralNetworkUtility import ActivationFunction

NUMBER_OF_GESTURES = 6
FOLDER_NAME = '../HackathonDataSamples/GestureData/'


SESSION_FOLDERS = '../HackathonDataSamples/NeuralNetwork/Sessions/'
TRAINING_DATA_FILE_PATH = '../HackathonDataSamples/NeuralNetwork/training_file.data'

SESS_PATH = SESSION_FOLDERS + '{}/'.format("2017-03-18-1617")

layer_sizes = [0, 150, 150, 150, 0]  # Network build
layer_activation_functions = [ActivationFunction.SIGMOID, ActivationFunction.SIGMOID]


class File:
    def __init__(self, filename, path):
        self.filename = filename
        self.path = path
        (self.gesture, self.example_id) = self.get_data_from_filename(filename)

    def get_file_path(self):
        return self.path + self.filename

    def get_data_from_filename(self, filename):
        pattern = r'Gesture([1-6])_Example([0-9]+).CSV'
        matchObj = re.match(pattern, filename, re.I)

        return (int(matchObj.group(1)) - 1, int(matchObj.group(2)))


class HackathonSamplesDataHandler(DataHandlers.DataHandler):
    def __init__(self, file):
        self.emg_data = numpy.genfromtxt(file.get_file_path(), delimiter=',')


def set_sess_path(session_name):
    global SESS_PATH
    SESS_PATH = SESSION_FOLDERS + "{}/".format(session_name)


def set_default_sess_path():
    set_sess_path(os.listdir(SESSION_FOLDERS)[-1])


def select_sess_path():
    session_folder_list = os.listdir(SESSION_FOLDERS)

    for i in range(len(session_folder_list)):
        print("{})".format(i), session_folder_list[i])
    session_choice = input("Select a session to use: ")
    try:
        session_choice = int(session_choice)
    except ValueError:
        session_choice = -1

    if session_choice >= len(session_folder_list) or session_choice < 0:
        return

    set_sess_path(session_folder_list[int(session_choice)])


def create_emg_training_file():
    file_list = []
    folder = FOLDER_NAME
    for filename in os.listdir(folder):
        if filename == ".gitignore":
            continue

        file = File(filename, folder)

        if file.example_id <= 1500 and file.gesture < NUMBER_OF_GESTURES:
            file_list.append(file)

    data_handler = HackathonSamplesDataHandler(file)
    n_input_nodes = len(data_handler.get_emg_data_features())
    NeuralNetworkUtility.create_emg_training_file(n_input_nodes, TRAINING_DATA_FILE_PATH, file_list, NUMBER_OF_GESTURES, HackathonSamplesDataHandler)


def create_emg_network():
    global SESS_PATH
    SESS_PATH = NeuralNetworkUtility.create_emg_network(SESSION_FOLDERS, layer_sizes, layer_activation_functions, TRAINING_DATA_FILE_PATH)


def train_emg_network():
    NeuralNetworkUtility.train_emg_network(TRAINING_DATA_FILE_PATH, SESS_PATH)


def test_emg_network():
    print("Session path:", SESS_PATH)
    file_list = []
    summary_list = []

    folder = FOLDER_NAME
    for filename in os.listdir(folder):
        if filename == ".gitignore":
            continue
        file = File(filename, folder)

        if file.example_id > 1500 and file.gesture < NUMBER_OF_GESTURES:
            file_list.append(file)

    for test_file in file_list:
        data_handler = HackathonSamplesDataHandler(test_file)

        start_time = time.time()
        results = NeuralNetworkUtility.input_test_emg_network(data_handler, SESS_PATH)
        end_time = time.time()

        recognized_gesture = numpy.argmax(results)
        print_results(results)

        print("Correct gesture:", test_file.gesture)
        print("Analyse time: ", "%.2f" % float(end_time - start_time))

        summary_list.append((test_file.gesture, recognized_gesture))

        print()
        print("File:", test_file.filename)

    print("#############################################################")
    print("Session path:", SESS_PATH)
    print("Summary List")

    success_list = []
    for i in range(NUMBER_OF_GESTURES):
        success_list.append([0, 0])

    for correct_gesture, recognized_gesture in summary_list:

        success_list[correct_gesture][0] += 1

        if correct_gesture == recognized_gesture:
            success_list[correct_gesture][1] += 1

        print(correct_gesture, " -> ", recognized_gesture)

    print()
    print("#############################################################")
    print("Success Rate")
    for i in range(NUMBER_OF_GESTURES):
        print('{:d}\t{:4d} of {:4d}'.format(i, success_list[i][1], success_list[i][0]))


def print_results(results):
    for result in results:
        print("\n###########################################################")
        for gesture in range(NUMBER_OF_GESTURES):
            print('{:d}\t{:10f}'.format(gesture, result[gesture]))

    print()
    print("Recognized: " + str(numpy.argmax(results)))
