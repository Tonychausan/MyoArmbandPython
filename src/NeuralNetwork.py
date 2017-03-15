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

# Constants
NEURAL_NETWORK_DATA_PATH = Constant.DATA_SET_FOLDER + "nn_data/"
SESSIONS_FOLDER_NAME = "sessions/"
LOG_FOLDER_NAME = "log/"

EMG_NEURAL_NETWORK_DATA_PATH = NEURAL_NETWORK_DATA_PATH + "emg_network/"
EMG_NEURAL_NETWORK_SESSIONS_FOLDER = EMG_NEURAL_NETWORK_DATA_PATH + SESSIONS_FOLDER_NAME

TRAINING_DATA_FILE_PATH = EMG_NEURAL_NETWORK_DATA_PATH + "training_file.data"

SESS_PATH = EMG_NEURAL_NETWORK_SESSIONS_FOLDER + "{}/".format("2017-03-13-0138")
SESS_MODEL_PATH = SESS_PATH + "emg_model"

# Training Parameters
N_EPOCH = 5000
LEARNING_RATE = 0.05

n_output_nodes = Constant.NUMBER_OF_GESTURES

layer_sizes = [0, 3 * 8, 8, n_output_nodes]  # Network build

tf.Session()  # remove warnings... hack...


def select_sess_path():
    global SESS_PATH
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
    SESS_PATH = EMG_NEURAL_NETWORK_SESSIONS_FOLDER + "{}/".format(session_folder_list[int(session_choice)])


def create_emg_training_file():
    print("Creating EMG-training file")
    data_handler = DataHandlers.FileDataHandler(DataUtility.TRAINING_FILE_LIST[0])
    n_inputs_nodes = len(data_handler.get_emg_data_features())

    with open(TRAINING_DATA_FILE_PATH, 'w') as outfile:
        outfile.write("{} ".format(DataUtility.SIZE_OF_TRAINING_FILE_LIST))
        outfile.write("{} ".format(n_inputs_nodes))
        outfile.write("{}\n".format(n_output_nodes))

        for i in range(DataUtility.SIZE_OF_TRAINING_FILE_LIST):
            data_file = DataUtility.TRAINING_FILE_LIST[i]
            print("Progress: {}%".format(int(((i + 1) / DataUtility.SIZE_OF_TRAINING_FILE_LIST) * 100)), end="\r")
            data_handler = DataHandlers.FileDataHandler(data_file)

            emg_sums = data_handler.get_emg_data_features()
            for i in range(n_inputs_nodes):
                outfile.write(str(emg_sums[i]))
                if i < n_inputs_nodes - 1:
                    outfile.write(" ")
                else:
                    outfile.write("\n")

            for gesture in range(n_output_nodes):
                if gesture != data_file.gesture:
                    outfile.write("0")
                else:
                    outfile.write("1")

                if gesture < Constant.NUMBER_OF_GESTURES - 1:
                    outfile.write(" ")
                else:
                    outfile.write("\n")

    print()
    print("Finished")
    print()
    print("training size:", DataUtility.SIZE_OF_TRAINING_FILE_LIST)
    print("Number of input neurons:", n_inputs_nodes)
    print("Number of output neurons:", n_output_nodes)
    print()


def create_network_meta_data_file(path):
    file_path = path + "network.meta"
    with open(file_path, 'w') as outfile:
        outfile.write("layer_sizes ")
        for layer_size in layer_sizes:
            outfile.write("{} ".format(layer_size))
        outfile.write("\n")

        outfile.write("Epoch_count 0")


def get_network_meta_data():
    file_path = SESS_PATH + "network.meta"
    with open(file_path, 'r') as metafile:
        layer_size_list = metafile.readline().split()[1:]
        epoch_count = int(metafile.readline().split(" ")[1])

    return (list(map(int, layer_size_list)), epoch_count)


def update_epoch_count_network_meta_data(epoch_count):
    file_path = SESS_PATH + "network.meta"

    with open(file_path, 'r') as metafile:
        lines = metafile.readlines()

    lines[1] = "Epoch_count {}".format(epoch_count)
    with open(file_path, 'w') as metafile:
        for line in lines:
            metafile.write(line)


def create_emg_network_variables(number_of_neuron_for_layer):
    number_of_variables = len(number_of_neuron_for_layer) - 1
    return_variables = []
    bias_variables = []

    for i in range(number_of_variables):
        variable_name = "theta{}".format(i)
        variable = tf.Variable(tf.random_uniform([number_of_neuron_for_layer[i], number_of_neuron_for_layer[i + 1]], -1, 1), name=variable_name)
        return_variables.append(variable)

        bias_name = "bias{}".format(i)
        bias = tf.Variable(tf.zeros(number_of_neuron_for_layer[i + 1]), name=bias_name)
        bias_variables.append(bias)

    return (return_variables, bias_variables)


def create_emg_network_layers(input_placeholder, variables, bias_variables):
    layers = []
    current_layer = input_placeholder
    for theta, bias in zip(variables, bias_variables):
        layer = tf.sigmoid(tf.matmul(current_layer, theta) + bias)
        layers.append(layer)
        current_layer = layer

    output = layers.pop()

    return (layers, output)


def get_training_inputs_and_outputs():
    inputs = []
    outputs = []

    with open(TRAINING_DATA_FILE_PATH, 'r') as training_data_file:
        (training_size, n_inputs, n_outputs) = training_data_file.readline().split()

        line_counter = 0
        for line in training_data_file:
            if line_counter % 2 == 0:
                inputs.append([float(x) for x in line.split()])
            else:
                outputs.append([float(x) for x in line.split()])

            line_counter += 1

    return (inputs, outputs)


def get_training_meta_data():
    with open(TRAINING_DATA_FILE_PATH, 'r') as training_data_file:
        (training_size, n_inputs, n_outputs) = training_data_file.readline().split()

    return(int(training_size), int(n_inputs), int(n_outputs))


def create_emg_network():
    sess_path = Constant.EMG_NEURAL_NETWORK_SESSIONS_FOLDER + "{}/".format(time.strftime("%Y-%m-%d-%H%M"))
    if os.path.exists(sess_path):
        run = input("A session with this name already exist, replace it? (y/n): ")
        if not run == "y":
            return

    print("Creating EMG-network")
    (inputs, outputs) = get_training_inputs_and_outputs()

    (training_size, n_inputs, n_outputs) = get_training_meta_data()

    input_placeholder = tf.placeholder(tf.float32, shape=[training_size, n_inputs], name="input")

    layer_sizes[0] = n_inputs
    layer_sizes[-1] = n_outputs

    (theta, bias) = create_emg_network_variables(layer_sizes)
    (layers, output) = create_emg_network_layers(input_placeholder, theta, bias)

    cost = tf.reduce_mean(tf.square(outputs - output))
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    saver = tf.train.Saver()

    if not os.path.exists(sess_path):
        os.makedirs(sess_path)

    sess_model_path = sess_path + "emg_model"
    saver.save(sess, sess_model_path)
    create_network_meta_data_file(sess_path)  # Write meta data of session to file

    print("EMG-network created")
    print("Session path:", sess_model_path)
    tf.reset_default_graph()


def print_training_info():
    os.symtem('cls')
    print("Train Network")
    print("Training file:", TRAINING_DATA_FILE_PATH)
    print("Training session:", SESS_PATH)


def train_emg_network():
    print_training_info()

    (inputs, outputs) = get_training_inputs_and_outputs()
    (training_size, n_inputs, n_outputs) = get_training_meta_data()
    (sess_layer_sizes, old_epoch_count) = get_network_meta_data()

    if(n_inputs != sess_layer_sizes[0] or n_outputs != sess_layer_sizes[-1]):
        print("Training file and session is not compatible!")
        return

    dummy = False
    while not dummy:
        n_steps = input("Number of steps: ")
        dummy = Utility.is_int_input(n_steps)
    n_steps = int(n_steps)

    dummy = False
    while not dummy:
        run_time = input("Max Run-time (hours): ")
        dummy = Utility.is_float_input(run_time)
    run_time = float(run_time) * 3600

    start_time = time.time()
    current_time = time.time() - start_time

    i = 0
    number_of_save = 0
    while current_time < run_time and i < n_steps:
        continue_emg_network_training(sess_layer_sizes, inputs, outputs, n_inputs, n_outputs, training_size, n_steps, i)

        print_training_info()
        print("Number of steps:", n_steps)
        print("Max Time (hours):", run_time / 3600)
        print()

        number_of_save += 1
        print("Number of save:", number_of_save)

        if i + N_EPOCH <= n_steps:
            i += N_EPOCH
        else:
            i += (n_steps % N_EPOCH)

        current_time = time.time() - start_time
        (hours, minutes, seconds) = Utility.second_to_HMS(current_time)
        print('Current time: {:.0f}h {:.0f}min {:.0f}sec'.format(hours, minutes, seconds))

        if i == 0:
            estimated_time = 0
        else:
            estimated_time = (current_time / i) * (n_steps)
        (hours, minutes, seconds) = Utility.second_to_HMS(estimated_time)
        print('Estimated time: {:.0f}h {:.0f}min {:.0f}sec'.format(hours, minutes, seconds))

        print('Batch:', i)
        update_epoch_count_network_meta_data(old_epoch_count + i)

    print()
    print("Runtime:", "{0:.2f}".format(float(time.time() - start_time)) + "sec")
    print("finished")


def continue_emg_network_training(sess_layer_sizes, inputs, outputs, n_inputs, n_outputs, training_size, n_steps, epoch_count):
    input_placeholder = tf.placeholder(tf.float32, shape=[training_size, n_inputs], name="input")
    output_placeholder = tf.placeholder(tf.float32, shape=[training_size, n_outputs], name="output")

    (theta, bias) = create_emg_network_variables(sess_layer_sizes)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, SESS_MODEL_PATH)

        (layer, output) = create_emg_network_layers(input_placeholder, theta, bias)

        cost = tf.reduce_mean(tf.square(outputs - output))
        train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)

        for i in range(N_EPOCH):
            if epoch_count + i >= n_steps:
                break
            sess.run(train_step, feed_dict={input_placeholder: inputs, output_placeholder: outputs})

        saver.save(sess, SESS_MODEL_PATH)

    tf.reset_default_graph()


def test_emg_network():
    summary_list = []

    for test_file in DataUtility.TEST_FILE_LIST:
        data_handler = DataHandlers.FileDataHandler(test_file)

        start_time = time.time()
        results = input_test_emg_network(data_handler)
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
        print('{:15s}\t{:4d} of {:4d}'.format(Gesture.gesture_to_string(i), success_list[i][1], success_list[i][0]))


def input_test_emg_network(input_data_handler):
    test_inputs = [input_data_handler.get_emg_data_features()]

    sess_layer_sizes = get_network_meta_data()[0]
    input_placeholder = tf.placeholder(tf.float32, shape=[1, sess_layer_sizes[0]], name="input")

    (theta, bias) = create_emg_network_variables(sess_layer_sizes)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, SESS_MODEL_PATH)

        (layers, output) = create_emg_network_layers(input_placeholder, theta, bias)

        results = sess.run(output, feed_dict={input_placeholder: test_inputs})

    tf.reset_default_graph()
    return results


def print_results(results):
    for result in results:
        print()
        print("###########################################################")
        for gesture in range(Gesture.NUMBER_OF_GESTURES):
            print('{:15s}\t{:10f}'.format(Gesture.gesture_to_string(gesture), result[gesture]))

    print()
    print("Recognized:", Gesture.gesture_to_string(np.argmax(results)))
