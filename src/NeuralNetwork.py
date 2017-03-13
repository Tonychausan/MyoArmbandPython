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
size_of_training_set = len(DataUtility.TRAINING_FILE_LIST)
size_of_test_set = len(DataUtility.TEST_FILE_LIST)
TRAINING_DATA_FILE_PATH = '../data/nn_data/emg_network/training_file.data'

SESS_PATH = '../data/nn_data/emg_network/sessions/{}/'.format("2017-03-13-0138")
SESS_MODEL_PATH =  SESS_PATH + 'emg_model'

# Training Parameters
N_STEPS = 5000
N_EPOCH = 5000
LEARNING_RATE = 0.05

# Varibales for creating new network
N_INPUT_NODES = Constant.NUMBER_OF_EMG_ARRAYS * 2
N_HIDDEN_NODES = 24
N_OUTPUT_NODES  = Constant.NUMBER_OF_GESTURES

layer_sizes = [N_INPUT_NODES, 3*8, 8, N_OUTPUT_NODES] # Network build

tf.Session() # remove warnings... hack...

def create_emg_training_file():
    data_handler = DataHandlers.FileDataHandler(DataUtility.TEST_FILE_LIST[0])
    N_INPUT_NODES = len(data_handler.get_emg_data_features())

    print("Creating EMG-training file")
    print("training size:", size_of_test_set)
    print("Number of input neurons:", N_INPUT_NODES)
    print("Number of output neurons:", N_OUTPUT_NODES)
    print()

    with open(TRAINING_DATA_FILE_PATH, 'w') as outfile:
        outfile.write(str(size_of_training_set) + " ")
        outfile.write(str(N_INPUT_NODES) + " ")
        outfile.write(str(N_OUTPUT_NODES) + "\n")

        for data_file in DataUtility.TRAINING_FILE_LIST:
            print(data_file.filename)
            data_handler = DataHandlers.FileDataHandler(data_file)

            emg_sums = data_handler.get_emg_data_features()
            for i in range(N_INPUT_NODES):
                outfile.write(str(emg_sums[i]))
                if i < N_INPUT_NODES - 1:
                    outfile.write(" ")
                else:
                    outfile.write("\n")

            for gesture in range(N_OUTPUT_NODES):
                if gesture != data_file.gesture:
                    outfile.write("0")
                else:
                    outfile.write("1")

                if gesture < Constant.NUMBER_OF_GESTURES - 1:
                    outfile.write(" ")
                else:
                    outfile.write("\n")

def create_network_meta_data_file(path):
    file_path = path + "network.meta"
    with open(file_path, 'w') as outfile:
        outfile.write("layer_sizes: ")
        for layer_size in layer_sizes:
            outfile.write(str(layer_size) + " ")

def get_network_meta_data_from_file():
    file_path = SESS_PATH + "network.meta"
    with open(file_path, 'r') as metafile:
        layer_size_list =  metafile.readline().split()[1:]

    return list(map(int, layer_size_list))


def create_emg_network_variables(number_of_neuron_for_layer):
    number_of_variables = len(number_of_neuron_for_layer) - 1
    return_variables = []
    bias_variables = []

    for i in range(number_of_variables):
        variable_name = "theta" + str(i)
        variable = tf.Variable(tf.random_uniform([number_of_neuron_for_layer[i], number_of_neuron_for_layer[i+1]], -1, 1), name=variable_name)
        return_variables.append(variable)

        bias_name = "bias" + str(i)
        bias = tf.Variable(tf.zeros(number_of_neuron_for_layer[i+1]), name=bias_name)
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
    sess_path = '../data/nn_data/emg_network/sessions/{}/'.format(time.strftime("%Y-%m-%d-%H%M"))
    if os.path.exists(sess_path):
        run_or_not = input("A session with this name already exist, replace it? (y/n): " )
        if not run_or_not == "y":
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

    sess_model_path = sess_path + 'emg_model'
    saver.save(sess, sess_model_path)
    create_network_meta_data_file(sess_path) #Write meta data of session to file

    print("EMG-network created")
    print("Session path:", sess_model_path)
    tf.reset_default_graph()

def continue_emg_network_training():
    print("Train Network")
    print("Training file:", TRAINING_DATA_FILE_PATH)
    (inputs, outputs) = get_training_inputs_and_outputs()

    (training_size, n_inputs, n_outputs) = get_training_meta_data()
    sess_layer_sizes = get_network_meta_data_from_file()

    if(n_inputs != sess_layer_sizes[0] or n_outputs != sess_layer_sizes[-1]):
        print("Training file and session is not compatible!")
        return

    print("Training session:", SESS_PATH)

    input_placeholder = tf.placeholder(tf.float32, shape=[training_size, n_inputs], name="input")
    output_placeholder = tf.placeholder(tf.float32, shape=[training_size, n_outputs], name="output")

    (theta, bias) = create_emg_network_variables(sess_layer_sizes)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, SESS_MODEL_PATH)

        (layer, output) = create_emg_network_layers(input_placeholder, theta, bias)

        cost = tf.reduce_mean(tf.square(outputs - output))
        train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)

        dummy = False
        while not dummy:
            n_steps = input("Number of steps: ")
            dummy = Utility.check_int_input(n_steps)
        n_steps = int(n_steps)

        dummy = False
        while not dummy:
            run_time = input("Max Time (hours): ")
            dummy = Utility.check_int_input(run_time)
        run_time = float(run_time) * 3600

        start_time = time.time()
        current_time = time.time() - start_time
        i = 0
        #for i in range(n_steps):
        while current_time < run_time and i < n_steps:
            sess.run(train_step, feed_dict={input_placeholder: inputs, output_placeholder: outputs})
            if i % N_EPOCH == 0:
                os.system('cls')
                print("Train Network")
                print("Training file:", TRAINING_DATA_FILE_PATH)
                print("Training session:", SESS_PATH)
                print("Number of steps:", n_steps)
                print("Max Time (hours):", run_time/3600)

                print('Batch:', i)
                if i != 0:
                    current_time = time.time() - start_time
                    (hours, minutes, seconds) = Utility.second_to_HMS(current_time)
                    print('Current time: {:.0f}h {:.0f}min {:.0f}sec'.format(hours, minutes, seconds))

                    estimated_time = (current_time/i) * (n_steps)
                    (hours, minutes, seconds) = Utility.second_to_HMS(estimated_time)
                    print('Estimated time: {:.0f}h {:.0f}min {:.0f}sec'.format(hours, minutes, seconds))

            i += 1

        print()
        print("Runtime:", '{0:.2f}'.format(float(time.time() - start_time))+"sec")
        print("finished")
        saver.save(sess, SESS_MODEL_PATH)

    tf.reset_default_graph()

def test_emg_network():
    test_inputs = []
    summary_list = []

    for test_file in DataUtility.TEST_FILE_LIST:
        data_handler = DataHandlers.FileDataHandler(test_file)

        start_time = time.time()
        results = input_test_emg_network(data_handler)
        end_time = time.time()

        recognized_gesture = np.argmax(results)
        print_results(results)

        print("Correct gesture:", Gesture.gesture_to_string(test_file.gesture))
        print("Analyse time: ", "%.2f"%float(end_time - start_time))

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

    sess_layer_sizes = get_network_meta_data_from_file()
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
        print("\n###########################################################")
        for gesture in range(Gesture.NUMBER_OF_GESTURES):
            print('{:15s}\t{:10f}'.format(Gesture.gesture_to_string(gesture), result[gesture]))

    print()
    print("Recognized: " + Gesture.gesture_to_string(np.argmax(results)))
