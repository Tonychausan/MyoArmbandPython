import numpy
import copy

import Utility
from DataUtility import Gesture


class ResultJsonName:
    FILENAME = "filename"
    RESULTS = "results"
    GESTURE = "gesture"


def print_results(number_of_gesture, filename, results, correct_gesture):
    print("###########################################################")
    print("File:", filename)
    for gesture in range(number_of_gesture):
        print('{:15s}\t{:10f}'.format(Gesture.gesture_to_string(gesture), results[gesture]))

    print()
    print("Correct:", Gesture.gesture_to_string(correct_gesture))


def print_success_rate(success_list):
    print("###########################################################")
    print("Success Rate")
    success_sum = [0, 0]
    for i in range(len(success_list)):
        if success_list[i][0] != 0:
            print('{:15s}\t{:4d} of {:4d} -> {:.2f}'.format(Gesture.gesture_to_string(i), success_list[i][1], success_list[i][0], 100 * success_list[i][1] / success_list[i][0]))
        success_sum[0] += success_list[i][1]
        success_sum[1] += success_list[i][0]

    print()
    print("Success rate: {:.2f}".format(100 * success_sum[0] / success_sum[1]))
    return (100 * success_sum[0] / success_sum[1])


def get_data_from_json_object(json_obj):
    filename = json_obj[ResultJsonName.FILENAME]
    results = json_obj[ResultJsonName.RESULTS]
    correct_gesture = json_obj[ResultJsonName.GESTURE]

    return (filename, correct_gesture, results)


def raw_success_list(number_of_gesture, json_data):
    success_list = []
    for i in range(number_of_gesture):
        success_list.append([0, 0])

    for result_object in json_data:
        (filename, correct_gesture, results) = get_data_from_json_object(result_object)
        recognized_gesture = numpy.argmax(results)

        success_list[correct_gesture][0] += 1
        if correct_gesture == recognized_gesture:
            success_list[correct_gesture][1] += 1
        else:
            print_results(number_of_gesture, filename, results, correct_gesture)
            print("Recognized:", Gesture.gesture_to_string(recognized_gesture))
            print()

    print_success_rate(success_list)


def input_parameter(input_text, default_value, min_value, max_value, is_int=True):
    user_input = input(input_text)
    check_input_function = Utility.is_int_input
    if not is_int:
        check_input_function = Utility.is_float_input
    if check_input_function(user_input):
        if is_int:
            user_input = int(user_input)
        else:
            user_input = float(user_input)
        if user_input >= min_value and user_input <= max_value:
            return user_input

    return default_value


def filtered_analyse(number_of_gesture, json_data):
    success_list = []
    for i in range(number_of_gesture):
        success_list.append([0, 0])

    gesture_margin = input_parameter("Gesture margin (default = 1): ", 1, 1, number_of_gesture)
    diff_margin = input_parameter("Difference margin (default = 0.0): ", 0.0, 0.0, 1.0, is_int=False)
    value_treshold = input_parameter("Value treshold (default = 0.0): ", 0.0, 0.0, 1.0, is_int=False)

    for result_object in json_data:
        (filename, correct_gesture, results) = get_data_from_json_object(result_object)

        is_success = False

        copy_results = copy.deepcopy(results)
        possible_gesture_results = []
        possible_gestures = []
        for i in range(gesture_margin):
            possible_gesture = numpy.argmax(copy_results)
            possible_gestures.append(possible_gesture)

            possible_gesture_result = copy_results[possible_gestures[-1]]
            copy_results[possible_gestures[-1]] = -1
            possible_gesture_results.append(possible_gesture_result)

        while numpy.amin(possible_gesture_results) - numpy.amax(copy_results) < diff_margin:
            i = numpy.argmin(possible_gesture_results)
            possible_gesture_results.pop(i)
            possible_gestures.pop(i)
            if not len(possible_gesture_results):
                break

        i = 0
        number_of_possible_gestures = len(possible_gestures)
        while i < number_of_possible_gestures:
            if possible_gesture_results[i] >= value_treshold:
                i += 1
            else:
                possible_gesture_results.pop(i)
                possible_gestures.pop(i)
                number_of_possible_gestures -= 1

        for i in range(number_of_possible_gestures):
            if possible_gesture_results[i] > value_treshold and possible_gestures[i] == correct_gesture:
                is_success = True

        success_list[correct_gesture][0] += 1
        if is_success:
            success_list[correct_gesture][1] += 1
        else:
            print_results(number_of_gesture, filename, results, correct_gesture)
            print("Recognized:", [Gesture.gesture_to_string(x) for x in possible_gestures])

    success_rate = print_success_rate(success_list)
    print("${}$ & ${:.2f}$ & ${:.2f}$ & ${:.2f}$".format(gesture_margin, diff_margin, value_treshold, success_rate))
