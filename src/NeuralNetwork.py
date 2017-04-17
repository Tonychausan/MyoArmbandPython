import tensorflow as tf

import Constants as Constant
import DataHandlers
from DataUtility import Gesture


NUMBER_OF_GESTURES = Gesture.NUMBER_OF_GESTURES
SESSION_FOLDERS = Constant.DATA_SET_FOLDER + "nn_data/emg_network/sessions/"


DATA_HANDLER_TYPE = DataHandlers.FileDataHandler


#tf.Session()  # remove warnings... hack...
