import numpy
import os
import time
import tensorflow as tf
import re

import DataHandlers
import NeuralNetworkUtility

NUMBER_OF_GESTURES = 6
FOLDER_NAME = '../HackathonDataSamples/GestureData/'
SESSION_FOLDERS = '../HackathonDataSamples/NeuralNetwork/Sessions/'


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


DATA_HANDLER_TYPE = HackathonSamplesDataHandler


def get_training_file_list(number_of_gestures):
    file_list = []
    folder = FOLDER_NAME
    for filename in os.listdir(folder):
        if filename == ".gitignore":
            continue

        file = File(filename, folder)

        # if file.example_id <= 1500 and file.gesture < number_of_gestures:
        if file.example_id > 1500 and file.gesture < number_of_gestures:
            file_list.append(file)

    return file_list


def get_test_file_list(number_of_gestures):
    file_list = []
    folder = FOLDER_NAME
    for filename in os.listdir(folder):
        if filename == ".gitignore":
            continue

        file = File(filename, folder)

        # if file.example_id > 1500 and file.gesture < number_of_gestures:
        if file.example_id <= 1500 and file.gesture < number_of_gestures:
            file_list.append(file)

    return file_list
