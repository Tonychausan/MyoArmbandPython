import tensorflow as tf
import numpy as np
import time
import datetime
import os
import numpy


import Utility
import DataHandlers
import HackathonDataNeuralNetwork
import DataUtility
from DataUtility import Sensor, Gesture, DataSetFormat, DataSetType, File


# Training Parameters
N_EPOCH = 5000
learning_rate = 0.05


class ActivationFunction:
    SIGMOID = "Sigmoid"
    RELU = "ReLu"
    SOFTMAX = "Softmax"


class NeuralNetwork:
    def __init__(self, session_folder, data_handler_type, is_hackathon):
        self.epoch_count = 0

        self.session_folder = session_folder
        self.data_handler_type = data_handler_type
        self.is_hackathon = is_hackathon

        self.set_sess_path(self.session_folder + os.listdir(self.session_folder)[-1])
        self.set_layer_sizes([])
        self.set_layer_activation_functions([])

    def get_number_of_gesture(self):
        if self.is_hackathon:
            return HackathonDataNeuralNetwork.NUMBER_OF_GESTURES
        else:
            return Gesture.NUMBER_OF_GESTURES

    def set_sess_path(self, sess_path):
        self.sess_path = sess_path + "/"
        self.file_path = self.sess_path + "network.meta"
        self.sess_model_path = self.sess_path + "emg_model"

    def set_layer_sizes(self, layer_sizes):
        self.layer_sizes = layer_sizes

    def set_layer_activation_functions(self, layer_activation_functions):
        self.layer_activation_functions = layer_activation_functions

    def select_sess_path(self):
        session_folder_list = os.listdir(self.session_folder)

        for i in range(len(session_folder_list)):
            print("{})".format(i), session_folder_list[i])
        session_choice = input("Select a session to use: ")
        try:
            session_choice = int(session_choice)
        except ValueError:
            session_choice = -1

        if session_choice >= len(session_folder_list) or session_choice < 0:
            return

        self.set_sess_path(session_folder_list[int(session_choice)])

    def create_network_meta_data_file(self):
        file_path = self.sess_path + "network.meta"
        with open(file_path, 'w') as outfile:
            outfile.write("layer_sizes ")
            for layer_size in self.layer_sizes:
                outfile.write("{} ".format(layer_size))
            outfile.write("\n")

            outfile.write("Epoch_count 0\n")

            outfile.write("layer_activation_functions ")
            for activation_function in self.layer_activation_functions:
                outfile.write("{} ".format(activation_function))
            outfile.write("\n")

    def get_network_meta_data(self):
        with open(self.file_path, 'r') as metafile:
            self.set_layer_sizes([int(x) for x in metafile.readline().split()[1:]])
            self.epoch_count = int(metafile.readline().split(" ")[1])
            self.set_layer_activation_functions(metafile.readline().split()[1:])

        return (list(map(int, self.layer_sizes)), self.layer_activation_functions, self.epoch_count)

    def update_epoch_count_network_meta_data(self, epoch_count):
        self.epoch_count = epoch_count
        with open(self.file_path, 'r') as metafile:
            lines = metafile.readlines()

        lines[1] = "Epoch_count {}\n".format(self.epoch_count)
        with open(self.file_path, 'w') as metafile:
            for line in lines:
                metafile.write(line)

    def create_emg_network_variables(self):
        number_of_variables = len(self.layer_sizes) - 1
        return_variables = []
        bias_variables = []

        for i in range(number_of_variables):
            variable_name = "theta{}".format(i)
            variable = tf.Variable(tf.random_uniform([self.layer_sizes[i], self.layer_sizes[i + 1]], -1, 1), name=variable_name)
            return_variables.append(variable)

            bias_name = "bias{}".format(i)
            bias = tf.Variable(tf.zeros(self.layer_sizes[i + 1]), name=bias_name)
            bias_variables.append(bias)

        return (return_variables, bias_variables)

    def create_emg_network_layers(self, input_placeholder, variables, bias_variables):
        layers = []
        current_layer = input_placeholder
        number_of_variables = len(variables)
        for i in range(number_of_variables):
            theta = variables[i]
            bias = bias_variables[i]

            activation_function = self.layer_activation_functions[i]
            if activation_function == ActivationFunction.SIGMOID:
                layer = tf.sigmoid(tf.matmul(current_layer, theta) + bias)
            elif activation_function == ActivationFunction.RELU:
                layer = tf.add(tf.matmul(current_layer, theta), bias)
                layer = tf.nn.relu(layer)
            elif activation_function == ActivationFunction.SOFTMAX:
                layer = tf.nn.softmax(tf.matmul(current_layer, theta) + bias)

            layers.append(layer)
            current_layer = layer

        output = layers.pop()

        return (layers, output)

    def get_training_inputs_and_outputs(self, training_file_path):
        inputs = []
        outputs = []

        with open(training_file_path, 'r') as training_data_file:
            (training_size, n_inputs, n_outputs) = training_data_file.readline().split()

            line_counter = 0
            for line in training_data_file:
                if line_counter % 2 == 0:
                    inputs.append([float(x) for x in line.split()])
                else:
                    outputs.append([float(x) for x in line.split()])

                line_counter += 1

        return (inputs, outputs)

    def get_training_meta_data(self, training_file_path):
        with open(training_file_path, 'r') as training_data_file:
            (training_size, n_inputs, n_outputs) = training_data_file.readline().split()

        return(int(training_size), int(n_inputs), int(n_outputs))

    def create_emg_training_file(self, file_list, training_file_path):
        data_handler = DataHandlers.FileDataHandler(DataUtility.TRAINING_FILE_LIST[0])
        n_input_nodes = len(data_handler.get_emg_data_features())

        if self.is_hackathon:
            n_output_nodes = HackathonDataNeuralNetwork.NUMBER_OF_GESTURES
        else:
            n_output_nodes = Gesture.NUMBER_OF_GESTURES
        size_of_training = len(file_list)

        with open(training_file_path, 'w') as outfile:
            outfile.write("{} ".format(size_of_training))
            outfile.write("{} ".format(n_input_nodes))
            outfile.write("{}\n".format(n_output_nodes))

            for i in range(size_of_training):
                data_file = file_list[i]
                print("Progress: {}%".format(int(((i + 1) / size_of_training) * 100)), end="\r")

                data_handler = self.data_handler_type(data_file)
                emg_sums = data_handler.get_emg_data_features()

                for i in range(n_input_nodes):
                    outfile.write(str(emg_sums[i]))
                    if i < n_input_nodes - 1:
                        outfile.write(" ")
                    else:
                        outfile.write("\n")

                for gesture in range(n_output_nodes):
                    if gesture != data_file.gesture:
                        outfile.write("0")
                    else:
                        outfile.write("1")

                    if gesture < n_output_nodes - 1:
                        outfile.write(" ")
                    else:
                        outfile.write("\n")

        print()
        print("Finished")
        print()
        print("training size:", size_of_training)
        print("Number of input neurons:", n_input_nodes)
        print("Number of output neurons:", n_output_nodes)
        print()

    def create_emg_network(self):
        sess_path_id = time.strftime("%Y-%m-%d-%H%M")
        new_sess_path = self.session_folder + "{}/".format(sess_path_id)
        self.set_sess_path(new_sess_path)

        if os.path.exists(self.sess_path):
            run = input("A session with this name already exist, replace it? (y/n): ")
            if not run == "y":
                return

        print("Create folder: {}".format(self.sess_path))
        if not os.path.exists(self.sess_path):
            os.makedirs(self.sess_path)

        print("Create EMG-training file")
        training_file_path = self.sess_path + "training_file.data"

        if not self.is_hackathon:
            file_list = DataUtility.TRAINING_FILE_LIST
        else:
            file_list = HackathonDataNeuralNetwork.get_training_file_list()
        self.create_emg_training_file(file_list, training_file_path)

        print("Create Network")
        number_of_hidden_layers = int(input("Input number of hidden layers: "))
        self.layer_sizes = [0] * (number_of_hidden_layers + 2)
        self.layer_activation_functions = [ActivationFunction.SIGMOID] * (number_of_hidden_layers + 1)
        print("Input the number of neurons for each hidden layer")

        for i in range(number_of_hidden_layers):
            hidden_layer_id = i + 1
            self.layer_sizes[hidden_layer_id] = int(input("Hidden layer {}: ".format(hidden_layer_id)))

        (inputs, outputs) = self.get_training_inputs_and_outputs(training_file_path)
        (training_size, n_inputs, n_outputs) = self.get_training_meta_data(training_file_path)

        input_placeholder = tf.placeholder(tf.float32, shape=[training_size, n_inputs], name="input")

        self.layer_sizes[0] = n_inputs
        self.layer_sizes[-1] = n_outputs

        (theta, bias) = self.create_emg_network_variables()
        (layers, output) = self.create_emg_network_layers(input_placeholder, theta, bias)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        saver = tf.train.Saver()

        saver.save(sess, self.sess_model_path)
        self.create_network_meta_data_file()  # Write meta data of session to file

        print("\n\nNetwork created")
        print("Session path:", self.sess_model_path)
        print("Layer sizes:", self.layer_sizes)
        print("Layer sizes:", self.layer_activation_functions)
        tf.reset_default_graph()

    def print_training_info(self, training_file_path):
        os.system('cls')
        print("Train Network")
        print("Training file:", training_file_path)
        print("Training session:", self.sess_path)

    def train_emg_network(self):
        training_file_path = self.sess_path + "training_file.data"
        self.print_training_info(training_file_path)

        (inputs, outputs) = self.get_training_inputs_and_outputs(training_file_path)
        (training_size, n_inputs, n_outputs) = self.get_training_meta_data(training_file_path)
        (sess_layer_sizes, layer_activation_functions, old_epoch_count) = self.get_network_meta_data()

        if(n_inputs != sess_layer_sizes[0] or n_outputs != sess_layer_sizes[-1]):
            print("Training file and session is not compatible!")
            return

        dummy = False
        while not dummy:
            n_steps = input("Number of steps: ")
            dummy = Utility.is_int_input(n_steps)
        n_steps = int(n_steps)

        start_time = time.time()
        current_time = time.time() - start_time

        i = 0
        global_step = old_epoch_count

        self.print_training_info(training_file_path)
        print("Number of steps:", n_steps)
        print('Current time: {:.0f}h {:.0f}min {:.0f}sec'.format(0, 0, 0))
        print('Estimated time: {:.0f}h {:.0f}min {:.0f}sec'.format(0, 0, 0))
        print('Batch:', global_step)
        print()

        while i < n_steps:
            self.continue_emg_network_training(inputs, outputs, n_inputs, n_outputs, training_size, n_steps, global_step)

            self.print_training_info(training_file_path)
            print("Number of steps:", n_steps)
            print()

            if global_step + N_EPOCH <= n_steps:
                global_step += N_EPOCH
                i += N_EPOCH
            else:
                global_step += (n_steps % N_EPOCH)
                i += (n_steps % N_EPOCH)

            current_time = time.time() - start_time
            (hours, minutes, seconds) = Utility.second_to_HMS(current_time)
            print('Current time: {:.0f}h {:.0f}min {:.0f}sec'.format(hours, minutes, seconds))

            if i == 0:
                estimated_time = 0
            else:
                estimated_time = (current_time / i) * (n_steps - old_epoch_count)
            (hours, minutes, seconds) = Utility.second_to_HMS(estimated_time)
            print('Estimated time: {:.0f}h {:.0f}min {:.0f}sec'.format(hours, minutes, seconds))

            print('Batch:', global_step)
            self.update_epoch_count_network_meta_data(global_step)

        print()
        print("Runtime:", "{0:.2f}".format(float(time.time() - start_time)) + "sec")
        print("finished")

    def continue_emg_network_training(self, inputs, outputs, n_inputs, n_outputs, training_size, n_steps, epoch_count):
        input_placeholder = tf.placeholder(tf.float32, shape=[training_size, n_inputs], name="input")
        output_placeholder = tf.placeholder(tf.float32, shape=[training_size, n_outputs], name="output")

        (theta, bias) = self.create_emg_network_variables()

        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, self.sess_model_path)

            (layer, output) = self.create_emg_network_layers(input_placeholder, theta, bias)

            # Mean Squared Estimate - the simplist cost function (MSE)
            cost = tf.reduce_mean(tf.square(outputs - output))
            train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

            for i in range(N_EPOCH):
                if epoch_count + i >= n_steps:
                    break
                sess.run(train_step, feed_dict={input_placeholder: inputs, output_placeholder: outputs})

            saver.save(sess, self.sess_model_path)

        tf.reset_default_graph()

    def test_emg_network(self):
        print("Session path:", self.sess_path)
        summary_list = []

        if not self.is_hackathon:
            file_list = DataUtility.TRAINING_FILE_LIST
        else:
            file_list = HackathonDataNeuralNetwork.get_test_file_list()

        for test_file in file_list:
            data_handler = self.data_handler_type(test_file)

            start_time = time.time()
            results = self.input_test_emg_network(data_handler)
            end_time = time.time()

            recognized_gesture = numpy.argmax(results)
            self.print_results(results)

            print("Correct gesture:", Gesture.gesture_to_string(test_file.gesture))
            print("Analyse time: ", "%.2f" % float(end_time - start_time))

            summary_list.append((test_file.gesture, recognized_gesture))

            print()
            print("File:", test_file.filename)

        print("#############################################################")
        print("Session path:", self.sess_path)
        print("Summary List")

        if self.is_hackathon:
            number_of_gestures = HackathonDataNeuralNetwork.NUMBER_OF_GESTURES
        else:
            number_of_gestures = Gesture.NUMBER_OF_GESTURES

        success_list = []
        for i in range(number_of_gestures):
            success_list.append([0, 0])

        for correct_gesture, recognized_gesture in summary_list:

            success_list[correct_gesture][0] += 1

            if correct_gesture == recognized_gesture:
                success_list[correct_gesture][1] += 1

            print(Gesture.gesture_to_string(correct_gesture), " -> ", Gesture.gesture_to_string(recognized_gesture))

        print()
        print("#############################################################")
        print("Success Rate")
        for i in range(number_of_gestures):
            print('{:15s}\t{:4d} of {:4d} -> {:.2f}'.format(Gesture.gesture_to_string(i), success_list[i][1], success_list[i][0], 100 * success_list[i][1] / success_list[i][0]))

    def input_test_emg_network(self, input_data_handler):
        test_inputs = [input_data_handler.get_emg_data_features()]

        self.get_network_meta_data()
        input_placeholder = tf.placeholder(tf.float32, shape=[1, self.layer_sizes[0]], name="input")

        (theta, bias) = self.create_emg_network_variables()

        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, self.sess_model_path)

            output = self.create_emg_network_layers(input_placeholder, theta, bias)[1]

            results = sess.run(output, feed_dict={input_placeholder: test_inputs})

        tf.reset_default_graph()
        return results

    def print_results(self, results):
        for result in results:
            print()
            print("###########################################################")
            for gesture in range(self.get_number_of_gesture()):
                print('{:15s}\t{:10f}'.format(Gesture.gesture_to_string(gesture), result[gesture]))

        print()
        print("Recognized:", Gesture.gesture_to_string(np.argmax(results)))
