import tensorflow as tf
import numpy as np
import time
import os
import numpy
import json


import Utility
import DataHandlers
import HackathonDataNeuralNetwork
import NeuralNetwork as NN
import DataUtility
from DataUtility import Gesture
import MenuUtility
import ResultAnalyses


# Training Parameters
N_EPOCH = 5000
learning_rate = 0.05


class ActivationFunction:
    SIGMOID = "Sigmoid"
    RELU = "ReLu"
    SOFTMAX = "Softmax"


class ResultJsonName:
    FILENAME = "filename"
    RESULTS = "results"
    GESTURE = "gesture"


class NeuralNetwork:
    def __init__(self, session_folder, data_handler_type, is_hackathon):
        self.epoch_count = 0

        self.session_folder = session_folder
        self.data_handler_type = data_handler_type
        self.is_hackathon = is_hackathon

        self.set_default_sess_path()
        self.get_network_meta_data()

    def set_default_sess_path(self):
        self.set_sess_path(os.listdir(self.session_folder)[-1])

    def change_dataset(self):
        if self.is_hackathon:
            self.is_hackathon = False
            DataFile = NN
        else:
            self.is_hackathon = True
            DataFile = HackathonDataNeuralNetwork

        self.session_folder = DataFile.SESSION_FOLDERS
        self.data_handler_type = DataFile.DATA_HANDLER_TYPE
        self.set_default_sess_path()

    def get_number_of_gesture(self):
        return self.number_of_gestures

    def set_sess_path(self, sess_path_id):
        self.sess_path = self.session_folder + "{}/".format(sess_path_id)
        self.file_path = self.sess_path + "network.meta"
        self.sess_model_path = self.sess_path + "emg_model"
        self.results_folder_path = self.sess_path + "results/"

    def set_layer_sizes(self, layer_sizes):
        self.layer_sizes = layer_sizes

    def set_layer_activation_functions(self, layer_activation_functions):
        self.layer_activation_functions = layer_activation_functions

    def print_sess_info(self, session_path):
        meta_data_path = session_path + "/" + "network.meta"
        with open(meta_data_path, 'r') as metafile:
            layer_sizes = metafile.readline().split()[1:]
            print("{:20s}".format("Number of gestures:"), layer_sizes[-1])
            print("{:20s}".format("Layer sizes:"), layer_sizes)
            print("{:20s}".format("Epoch count:"), int(metafile.readline().split(" ")[1]))
            print("{:20s}".format("Activations:"), metafile.readline().split()[1:])  # layer_activation_functions
            print("{:20s}".format("Wavelet level:"), int(metafile.readline().split(" ")[1]))
            print("{:20s}".format("[MAV, RMS, WL]:"), [int(x) for x in metafile.readline().split()[1:]])

    def select_sess_path(self):
        session_folder_list = os.listdir(self.session_folder)

        for i in range(len(session_folder_list)):
            print("{})".format(i), session_folder_list[i])
            self.print_sess_info(self.session_folder + session_folder_list[i])
            print()
        session_choice = input("Select a session to use: ")
        try:
            session_choice = int(session_choice)
        except ValueError:
            session_choice = -1

        if session_choice >= len(session_folder_list) or session_choice < 0:
            return

        self.set_sess_path(session_folder_list[int(session_choice)])
        self.get_network_meta_data()

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

            outfile.write("Wavelet_level {}\n".format(self.wavelet_level))

            outfile.write("Features ")
            for feature in self.feature_function_check_list:
                outfile.write("{} ".format(feature))
            outfile.write("\n")

    def get_network_meta_data(self):
        with open(self.file_path, 'r') as metafile:
            self.set_layer_sizes([int(x) for x in metafile.readline().split()[1:]])
            self.number_of_gestures = self.layer_sizes[-1]
            self.epoch_count = int(metafile.readline().split(" ")[1])
            self.set_layer_activation_functions(metafile.readline().split()[1:])

            self.wavelet_level = int(metafile.readline().split(" ")[1])
            self.feature_function_check_list = [int(x) for x in metafile.readline().split()[1:]]

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
        data_handler.set_emg_wavelet_level(self.wavelet_level)
        data_handler.set_feature_functions_list(self.feature_function_check_list)
        n_input_nodes = len(data_handler.get_emg_data_features())

        n_output_nodes = self.get_number_of_gesture()
        size_of_training = len(file_list)

        with open(training_file_path, 'w') as outfile:
            outfile.write("{} ".format(size_of_training))
            outfile.write("{} ".format(n_input_nodes))
            outfile.write("{}\n".format(n_output_nodes))

            for i in range(size_of_training):
                data_file = file_list[i]
                print("Training file progress: {}%".format(int(((i + 1) / size_of_training) * 100)), end="\r")

                data_handler = self.data_handler_type(data_file)
                data_handler.set_emg_wavelet_level(self.wavelet_level)
                data_handler.set_feature_functions_list(self.feature_function_check_list)
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
        # new_sess_path = self.session_folder + "{}/".format(sess_path_id)
        self.set_sess_path(sess_path_id)

        if os.path.exists(self.sess_path):
            run = input("A session with this name already exist, replace it? (y/n): ")
            if not run == "y":
                return

        print("Create folder: {}".format(self.sess_path))
        if not os.path.exists(self.sess_path):
            os.makedirs(self.sess_path)

        print("\nCreate EMG-training file")
        training_file_path = self.sess_path + "training_file.data"

        number_of_gestures = self.get_number_of_gesture()
        if not self.is_hackathon:
            file_list = DataUtility.TRAINING_FILE_LIST
        else:
            file_list = HackathonDataNeuralNetwork.get_training_file_list(number_of_gestures)

        number_of_gestures = input("Number of gestures: ")
        if Utility.is_int_input(number_of_gestures):
            self.number_of_gestures = int(number_of_gestures)

        wavelet_level = input("Use Wavelet Level: ")
        if Utility.is_int_input(wavelet_level):
            self.wavelet_level = int(wavelet_level)
        print()

        self.feature_function_check_list = [0, 0, 0]
        feature_name_list = ["Mean Absoulute Value", "Root Mean Square", "Waveform Length"]
        for i in range(len(feature_name_list)):
            use_feature = input("Use {} (y/n): ".format(feature_name_list[i]))
            if use_feature == 'y':
                self.feature_function_check_list[i] = 1
        print()

        self.create_emg_training_file(file_list, training_file_path)

        print("Create Network")
        number_of_hidden_layers = int(input("Number of hidden layers: "))
        print()
        self.layer_sizes = [0] * (number_of_hidden_layers + 2)
        self.layer_activation_functions = [ActivationFunction.SIGMOID] * (number_of_hidden_layers + 1)
        print("Number of neurons for each hidden layer")

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

        print("\n\nNetwork created")
        print("Session path:", self.sess_model_path)
        print("Layer sizes:", self.layer_sizes)
        print("Layer activation functions:", self.layer_activation_functions)
        tf.reset_default_graph()

        print("\nCreate meta-data file")
        self.create_network_meta_data_file()  # Write meta data of session to file
        self.print_sess_info(self.sess_path)

        input("\nPress Enter to continue...")

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

        while global_step < n_steps:
            self.continue_emg_network_training(inputs, outputs, n_inputs, n_outputs, training_size, n_steps, global_step)

            self.print_training_info(training_file_path)
            print("Number of steps:", n_steps)
            print()

            if global_step + N_EPOCH <= n_steps:
                global_step += N_EPOCH
                i += N_EPOCH
            else:
                global_step += ((n_steps - old_epoch_count) % N_EPOCH)
                i += ((n_steps - old_epoch_count) % N_EPOCH)

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
        input("Press Enter to continue...")

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
        self.get_network_meta_data()
        print("Session path:", self.sess_path)

        is_storeing_result = input("Write result to file (y/n)? ")
        if is_storeing_result == 'y':
            is_storeing_result = True
        else:
            is_storeing_result = False
        summary_list = []

        run_date = time.strftime("%Y-%m-%d-%H%M")

        number_of_gestures = self.get_number_of_gesture()
        if not self.is_hackathon:
            file_list = DataUtility.TEST_FILE_LIST
        else:
            file_list = HackathonDataNeuralNetwork.get_test_file_list(number_of_gestures)

        for test_file in file_list:
            data_handler = self.data_handler_type(test_file)
            data_handler.set_emg_wavelet_level(self.wavelet_level)
            data_handler.set_feature_functions_list(self.feature_function_check_list)

            start_time = time.time()
            results = self.input_test_emg_network(data_handler)
            end_time = time.time()

            recognized_gesture = numpy.argmax(results)

            print()
            print("###########################################################")
            self.print_results(results)
            print()
            print("Recognized:", Gesture.gesture_to_string(np.argmax(results)))

            print("Correct gesture:", Gesture.gesture_to_string(test_file.gesture))
            print("Analyse time: ", "%.2f" % float(end_time - start_time))

            summary_list.append((test_file.gesture, recognized_gesture))

            print()
            print("File:", test_file.filename)
            if is_storeing_result:
                self.write_result_to_file(results, test_file.filename, test_file.gesture, run_date)

        print("#############################################################")
        print("Session path:", self.sess_path)
        print("Summary List")

        number_of_gestures = self.get_number_of_gesture()

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
            if success_list[i][0] != 0:
                print('{:15s}\t{:4d} of {:4d} -> {:.2f}'.format(Gesture.gesture_to_string(i), success_list[i][1], success_list[i][0], 100 * success_list[i][1] / success_list[i][0]))

        input("Press Enter to continue...")

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
        return results[0]

    def print_results(self, results):
        for gesture in range(self.get_number_of_gesture()):
            print('{}) {:15s}\t{:10f}'.format(gesture, Gesture.gesture_to_string(gesture), results[gesture]))

    def write_result_to_file(self, results, file_name, correct_gesture, run_date):
        results_folder_path = self.results_folder_path
        json_file_path = results_folder_path + "raw_results_{}.json".format(run_date)

        if not os.path.exists(results_folder_path):
            os.makedirs(results_folder_path)

        if os.path.isfile(json_file_path):
            with open(json_file_path) as json_file:
                json_data = json.load(json_file)
        else:
            json_data = json.loads('[]')

        list_results = []
        list_results = results.tolist()

        json_object_result = '{{ "filename" : "{}", "gesture" : {}, "results" : {} }}'.format(file_name, correct_gesture, list_results)
        json_data.append(json.loads(json_object_result))

        with open(json_file_path, 'w') as outfile:
            json.dump(json_data, outfile, sort_keys=True, indent=4, separators=(',', ': '))

    def result_analyses(self):
        if not os.path.exists(self.results_folder_path):
            print("No results found!")
            input("Press enter to continue...")
            return

        result_file_list = os.listdir(self.results_folder_path)
        if len(result_file_list) == 0:
            print("No results found!")
            input("Press enter to continue...")
            return

        result_file_path = self.results_folder_path + result_file_list[-1]
        if len(result_file_list) > 1:
            for i in range(len(result_file_list)):
                print("{})".format(i), result_file_list[i])
            result_choice = input("Select a result file to use: ")
            try:
                if not (result_choice >= len(result_file_list) or result_choice < 0):
                    result_choice = int(result_choice)
                    result_file_path = self.results_folder_path + result_file_list[result_choice]
            except ValueError:
                pass

        print("Result file: {}".format(result_file_path))
        with open(result_file_path) as json_file:
            json_result_data = json.load(json_file)

        analyse_menu = [
            MenuUtility.MenuItem("Raw success list", ResultAnalyses.raw_success_list),
            MenuUtility.MenuItem("Filtered analyse", ResultAnalyses.filtered_analyse)
        ]

        print("Analyses menu")
        print("####################################################")
        action = MenuUtility.print_menu(analyse_menu)
        analyse_menu[action].function(self.get_number_of_gesture(), json_result_data)

        input("Press enter to continue...")
