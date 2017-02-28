from time import sleep
import myo as libmyo
from os import walk

import DeviceListener as DeviceListener
import MenuActions as MenuActions
import DataHandlers as DataHandlers
import DataUtility as DataUtility

from DataUtility import Sensor, Gesture, DataSetFormat, DataSetType, File
from MenuActions import MainMenuItem


menu_item_list = []

def menu():
    menu_item_list = MenuActions.create_main_menu()

    action = MenuActions.print_menu(menu_item_list)

    if not menu_item_list[action].is_myo_depended:
        menu_item_list[action].function()
    else:
        libmyo.init('../myo-sdk-win-0.9.0/bin')
        listener = DeviceListener.Listener()
        hub = libmyo.Hub()
        hub.run(200, listener)

        menu_item_list[action].function(listener)

        hub.shutdown()  # !! crucial

def main():
     menu()



main()
