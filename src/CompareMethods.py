import numpy
import sys

import Constants as Constant
import Utility
import DataHandlers
import DataUtility
from DataUtility import Sensor, Gesture, DataSetType
from Settings import is_sensor_on


def cross_correlation(max_delay, x, y):
    size_of_array = len(x)

    mx = numpy.mean(x)
    my = numpy.mean(y)

    temp_sx = numpy.subtract(x, mx)
    temp_sx = numpy.multiply(temp_sx, temp_sx)
    temp_sy = numpy.subtract(y, my)
    temp_sy = numpy.multiply(temp_sy, temp_sy)

    sx = numpy.sum(temp_sx)
    sy = numpy.sum(temp_sy)

    denom = numpy.sqrt(sx*sy)
    r = 0.0
    for delay in range(round(-max_delay + 1), round(max_delay)):
        sxy = 0
        for i in range(size_of_array):
            j = i + delay
            if j < 0 or j >= size_of_array:
                continue
            else:
                sxy += (x[i] - mx) * (y[j] - my)

        if r < (sxy/denom):
            r = sxy/denom

    return r


def cross_correlation_compare_sensor_data(input_data, check_data, sensor):
    number_of_arrays = Utility.get_number_of_arrays_for_sensor(sensor)
    data_length = Utility.get_length_of_arrays_for_sensor(sensor)

    r = 0.0;
    for i in range(number_of_arrays):
        r += cross_correlation(data_length/2, input_data[i], check_data[i])

    return r / number_of_arrays

def cross_correlation_comparison(gesture_input_data_handler):
    print("Sensor Ignored: ", end="")
    for sensor in range(Sensor.NUMBER_OF_SENSORS):
        if not is_sensor_on(sensor):
            print(Sensor.sensor_to_string(sensor) + ", ", end="")

    print("\b\b", "  ")
    print("Compare method: Cross Correlation\n")

    corr_rs = [0] * Constant.NUMBER_OF_GESTURES
    emg_corrs = [0] * Constant.NUMBER_OF_GESTURES

    for i in range(len(DataUtility.TRAINING_FILE_LIST)):
        training_data_file = DataUtility.TRAINING_FILE_LIST[i]
        print(training_data_file.filename, end="\r")
        sys.stdout.write("\033[K") # clean line

        current_training_gesture = training_data_file.gesture

        gesture_training_data_handler = DataHandlers.FileDataHandler(training_data_file)

        for sensor in range(Sensor.NUMBER_OF_SENSORS):
            if not is_sensor_on(sensor):
                continue
            elif sensor == Sensor.EMG:
                emg_corrs[current_training_gesture] += cross_correlation_compare_sensor_data(gesture_input_data_handler.get_sensor_data(sensor), gesture_training_data_handler.get_sensor_data(sensor), sensor)
            else:
                corr_rs[current_training_gesture] += cross_correlation_compare_sensor_data(gesture_input_data_handler.get_sensor_data(sensor), gesture_training_data_handler.get_sensor_data(sensor), sensor)

    recognized_gesture = Gesture.NONE_GESTURE
    max_similarity = 0.0
    for gesture in range(Gesture.NUMBER_OF_GESTURES):
        similarity = corr_rs[gesture] + emg_corrs[gesture]
        print("{:10s}".format(Gesture.gesture_to_string(gesture)) + ": r = ", end="")
        print("%.8f" % similarity, end="")

        print(", IMU = ", end="")
        print("%.8f" % corr_rs[gesture], end="")

        print(", EMG = ", end="")
        print("%.8f" % emg_corrs[gesture])

        if (similarity > max_similarity):
            max_similarity = similarity
            recognized_gesture = gesture

    print("")
    return recognized_gesture
