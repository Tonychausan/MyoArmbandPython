import tensorflow as tf
import numpy as np

import Constants as Constant
from DataUtility import Sensor, Gesture, DataSetFormat, DataSetType, File
import DataUtility as DataUtility
import DataHandlers as DataHandlers

# Constants
size_of_training_set = len(DataUtility.TRAINING_FILE_LIST)
size_of_test_set = len(DataUtility.TEST_FILE_LIST)
training_data_file_path = '../data/nn_data/emg_network/training_file.data'
sess_path = '../data/nn_data/emg_network/session/emg_model'


# Neural Network Parameters
N_STEPS = 500000
N_EPOCH = 5000

N_INPUT_NODES = Constant.NUMBER_OF_EMG_ARRAYS
N_HIDDEN_NODES = 16
N_OUTPUT_NODES  = Constant.NUMBER_OF_GESTURES
LEARNING_RATE = 0.05


def create_emg_training_file():
    print("Creating EMG-training file")
    with open(training_data_file_path, 'w') as outfile:
        outfile.write(str(size_of_training_set) + " ")
        outfile.write(str(N_INPUT_NODES) + " ")
        outfile.write(str(N_OUTPUT_NODES) + "\n")

        for data_file in DataUtility.TRAINING_FILE_LIST:
            print(data_file.filename)
            data_handler = DataHandlers.FileDataHandler(data_file)

            emg_sums = data_handler.get_emg_sums_normalized()
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

def create_emg_network_variables():
    theta1 = tf.Variable(tf.random_uniform([N_INPUT_NODES, N_HIDDEN_NODES], -1, 1), name="theta1")
    theta2 = tf.Variable(tf.random_uniform([N_HIDDEN_NODES, N_OUTPUT_NODES], -1, 1), name="theta2")

    bias1 = tf.Variable(tf.zeros([N_HIDDEN_NODES]), name="bias1")
    bias2 = tf.Variable(tf.zeros([N_OUTPUT_NODES]), name="bias2")
    return (theta1, theta2, bias1, bias2)

def create_emg_network_layers(input_placeholder, theta1, theta2, bias1, bias2):
    layer1 = tf.sigmoid(tf.matmul(input_placeholder, theta1) + bias1)
    output = tf.sigmoid(tf.matmul(layer1, theta2) + bias2)

    return (layer1, output)

def create_emg_network():
    inputs = []
    outputs = []

    with open(training_data_file_path, 'r') as training_data_file:
        (training_size, n_inputs, n_outputs) = training_data_file.readline().split()

        line_counter = 0
        for line in training_data_file:
            if line_counter % 2 == 0:
                inputs.append([float(x) for x in line.split()])
            else:
                outputs.append([float(x) for x in line.split()])

            line_counter += 1

    input_placeholder = tf.placeholder(tf.float32, shape=[training_size, N_INPUT_NODES], name="input")
    output_placeholder = tf.placeholder(tf.float32, shape=[training_size, N_OUTPUT_NODES], name="output")

    (theta1, theta2, bias1, bias2) = create_emg_network_variables()
    (layer1, output) = create_emg_network_layers(input_placeholder, theta1, theta2, bias1, bias2)

    cost = tf.reduce_mean(tf.square(outputs - output))
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    for i in range(N_STEPS):
        sess.run(train_step, feed_dict={input_placeholder: inputs, output_placeholder: outputs})
        if i % N_EPOCH == 0:
            print('Batch ', i)

    saver = tf.train.Saver()
    saver.save(sess, sess_path)

def continue_emg_training():
    inputs = []
    outputs = []

    with open(training_data_file_path, 'r') as training_data_file:
        (training_size, n_inputs, n_outputs) = training_data_file.readline().split()

        line_counter = 0
        for line in training_data_file:
            if line_counter % 2 == 0:
                inputs.append([float(x) for x in line.split()])
            else:
                outputs.append([float(x) for x in line.split()])

            line_counter += 1

    input_placeholder = tf.placeholder(tf.float32, shape=[training_size, N_INPUT_NODES], name="input")
    output_placeholder = tf.placeholder(tf.float32, shape=[training_size, N_OUTPUT_NODES], name="output")

    (theta1, theta2, bias1, bias2) = create_emg_network_variables()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, sess_path)

        (layer1, output) = create_emg_network_layers(input_placeholder, theta1, theta2, bias1, bias2)

        cost = tf.reduce_mean(tf.square(outputs - output))
        train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)

        for i in range(N_STEPS):
            sess.run(train_step, feed_dict={input_placeholder: inputs, output_placeholder: outputs})
            if i % N_EPOCH == 0:
                print('Batch ', i)

        saver.save(sess, sess_path)



def test_emg_network():
    test_inputs = []
    for test_file in DataUtility.TEST_FILE_LIST:
        data_handler = DataHandlers.FileDataHandler(test_file)
        test_inputs.append(data_handler.get_emg_sums_normalized())

    with open(training_data_file_path, 'r') as training_data_file:
        (training_size, n_inputs, n_outputs) = training_data_file.readline().split()

    input_placeholder = tf.placeholder(tf.float32, shape=[size_of_test_set, N_INPUT_NODES], name="input")

    (theta1, theta2, bias1, bias2) = create_emg_network_variables()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, sess_path)

        (layer1, output) = create_emg_network_layers(input_placeholder, theta1, theta2, bias1, bias2)

        results = sess.run(output, feed_dict={input_placeholder: test_inputs})

    for result, test_file in zip(results, DataUtility.TEST_FILE_LIST):
        print("\n###########################################################")
        for gesture in range(Gesture.NUMBER_OF_GESTURES):
            print(Gesture.gesture_to_string(gesture), result[gesture])

        print()
        print("Gesture: " + Gesture.gesture_to_string(test_file.gesture))
        print("Recognized: " + Gesture.gesture_to_string(np.argmax(result)))

    print("#############################################################")
    print("Summary List")
    for result, test_file in zip(results, DataUtility.TEST_FILE_LIST):
        print(Gesture.gesture_to_string(test_file.gesture), " -> ", Gesture.gesture_to_string(np.argmax(result)))


#continue_emg_training()
#create_emg_training_file()
#test_emg_network()
#create_emg_network()
