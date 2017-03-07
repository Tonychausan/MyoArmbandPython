from time import sleep
import myo as libmyo
from os import walk
import datetime

import DeviceListener as DeviceListener
import MenuActions as MenuActions
import Utility
import DataHandlers as DataHandlers
import DataUtility as DataUtility

from DataUtility import Sensor, Gesture, DataSetFormat, DataSetType, File

menu_item_list = []

def menu():
    menu_item_list = MenuActions.create_main_menu()
    action = MenuActions.print_menu(menu_item_list)
    menu_item_list[action].function()

def main():
    menu()

main()
