import numpy as np
import pandas as pd
import os.path
import time

COLUMN_TO_TYPE = {
    'acoustic_data': np.int16,
    'time_to_failure': np.float64

}

part1_directory = r'../column_files'


COLUMN_TO_FOLDER = {
    'acoustic_data': part1_directory,
    'time_to_failure': part1_directory
}



def init_reading():

    info = {
        "0": [0,   5656574],
        "1": [5656574,   50085878],
        "2": [50085878, 104677356],
        "3": [104677356, 138772453],
        "4": [138772453, 187641820],
        "5": [187641820, 218652630],
        "6": [218652630, 245829585],
        "7": [245829585, 307838917],
        "8": [307838917, 338276287],
        "9": [338276287, 375377848],
        "10": [375377848, 419368880],
        "11": [419368880, 461811623],
        "12": [461811623, 495800225],
        "13": [495800225, 528777115],
        "14": [528777115, 585568144],
        "15": [585568144, 621985673],
        "16": [621985673, 629145479]
    }

    mmaps = {}
    for column, dtype in COLUMN_TO_TYPE.items():
        directory = COLUMN_TO_FOLDER[column]
        file_path = os.path.join(directory, 'train_{}.bin'.format(column))
        mmap = np.memmap(file_path, dtype=COLUMN_TO_TYPE[column], mode='r', shape=(629145479,))
        mmaps[column] = mmap

    info['mmaps'] = mmaps

    return info



def read_object_info(info, object_id, as_pandas=True, columns=None):
    start = info[str(object_id)][0]
    end = info[str(object_id)][1]

    data = read_object_by_index_range(info, start, end, as_pandas, columns)
    return data


def read_object_by_index_range(info, start, end, as_pandas=True, columns=None):
    data = {}
    for column, mmap in info['mmaps'].items():
        if columns is None or column in columns:
            data[column] = mmap[start: end]

    if as_pandas:
        data = pd.DataFrame(data)

    return data