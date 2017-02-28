from time import sleep
import myo as libmyo
from os import walk

import DeviceListener as DeviceListener
import MenuActions as MenuActions
import DataHandlers as DataHandlers
import CompareMethods

from DataUtility import Sensor, Gesture, DataSetFormat, DataSetType, File
import DataUtility as DataUtility

def isMyoDependentActivity(action):
    action = action - 1
    myo_dependency_list = [True, True, False, False, False, False]
    return myo_dependency_list[action]

def print_menu():
    print( "1) Try gestures" )
    print( "2) Measurment display" )
    print( "3) Compress files" )
    print( "4) Delete all compressed files" )
    print( "5) Pre-data gesture comparison" )
    print( "6) Test Neural Network" )


def menu():
    print_menu()
    action = int(input( "Choose an action: " ))
    if not isMyoDependentActivity(action):
        if action == 3:
            MenuActions.compress_json_files()
        elif action == 4:
            MenuActions.remove_all_compressed_files()
        elif action == 5:
            MenuActions.compare_prerecorded_files()
        elif action == 6:
            print("Neural network")
    else:
        libmyo.init('../myo-sdk-win-0.9.0/bin')
        listener = DeviceListener.Listener()
        hub = libmyo.Hub()
        hub.run(200, listener)

        if action == 1:
            print("Try Gesture")
            MenuActions.live_gesture_recognition()
        elif action == 2:
            MenuActions.print_myo_data(listener)

        hub.shutdown()  # !! crucial

def main():
     menu()



main()
