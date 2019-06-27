import time
import os.path
import numpy as np
import pandas as pd

COLUMN_TO_TYPE = {
    'acoustic_data': np.int16,
    'time_to_failure': np.float64

}

def prepare_data(directory, name, output_columns):
    start_time = time.time()
    file_path = os.path.join(directory, '{}.csv'.format(name))
    print('reading {}  '.format(file_path))

    dtypes = {column: COLUMN_TO_TYPE[column] for column in output_columns}

    data = pd.read_csv(file_path, usecols =output_columns, dtype=dtypes, engine='c')
    print("{:6.4f} secs".format((time.time() - start_time)))

    for column in output_columns:
        output_file_name = '{}_{}.bin'.format(name, column)
        print('dumping {}  '.format(output_file_name), end='')
        start_time = time.time()
        mmap = np.memmap(output_file_name, dtype=COLUMN_TO_TYPE[column], mode='w+', shape=(data.shape[0]))
        mmap[:] = data[column].values
        del mmap
        print("{:6.4f} secs".format((time.time() - start_time)))


def main():
    directory = 'dataset'
    output_columns = ['acoustic_data', 'time_to_failure']
    prepare_data(directory, 'train', output_columns)


if __name__ == '__main__':
    main()