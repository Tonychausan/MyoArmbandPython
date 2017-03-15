from DataUtility import Sensor

# List of sensors to ignor in the gesture prediction
is_EMG_sensor_on = True
is_ACC_sensor_on = True
is_GYR_sensor_on = True
is_ORI_sensor_on = True

# DTW or cross-correlation
is_DTW_used = False


def is_sensor_on(sensor):
    if sensor == Sensor.EMG:
        return is_EMG_sensor_on
    elif sensor == Sensor.ACC:
        return is_ACC_sensor_on
    elif sensor == Sensor.GYR:
        return is_GYR_sensor_on
    elif sensor == Sensor.ORI:
        return is_ORI_sensor_on
    else:
        return []
