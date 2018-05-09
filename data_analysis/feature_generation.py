import os
from multiprocessing import cpu_count

import pandas as pd
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import roll_time_series
from data.data_loader import DataLoader

# Settings

# relevant features for tsfresh generation
FEATURES = [
    'node_id',
    'relative_timestamp',
    'temperature',
    'relative_humidity',
    'light_sensor_1_wvl_nm',
    'light_sensor_2_wvl_nm'
]

# base features that will be joined to the new feature matrix
JOIN_FEATURES = [
    'state_door',
    'state_window',
    'room_location',
    'measurement_no',
    'absolute_timestamp'
]

# size of windows in number of past observations (≈ 4 sec, ± 1 sec between observations)
WINDOW_SIZE = 30

# number of cores used for the feature computation
NCORES = cpu_count() - 1

loader = DataLoader(path=os.path.join(os.getcwd(), 'SmartHomeSensorics_full_data.csv'))
full_data = loader.raw_data
room_params = {'A': 60,
               'B': 44,
               'C': 40}

feature_matrix_list = []

for room in room_params:
    for measurement in range(1, room_params[room] + 1):
        print(f'Room: {room}, Measurement: {measurement}')
        subset_data = loader.return_experiment_measurement(room_location=room, measurement_no=measurement, sensor_node=1)
        subset_data.sort_values(['node_id', 'relative_timestamp'], inplace=True)
        ts_for_rolling = subset_data[FEATURES]
        X_roll = roll_time_series(ts_for_rolling, column_id='node_id', column_sort='relative_timestamp',
                                  column_kind=None, rolling_direction=1, max_timeshift=WINDOW_SIZE)

        # for timepoint in X_roll['node_id'].unique():
        # print(f"Timepoint: {timepoint}, Subset size: {X_roll.loc[X_roll.node_id == timepoint].size}")
        temp_df = extract_features(X_roll, n_jobs=NCORES, column_sort='relative_timestamp',
                                   column_id='node_id', column_kind=None, show_warnings=False)
        join_df = subset_data[FEATURES + JOIN_FEATURES]
        join_df.set_index('relative_timestamp', inplace=True)
        join_df = join_df.join(temp_df)
        feature_matrix_list.append(join_df)

full_feature_matrix = pd.concat(feature_matrix_list, axis=0, ignore_index=True)
full_feature_matrix = full_feature_matrix.reset_index(drop=True)
full_feature_matrix.to_csv('full_feature_matrix.csv', index=False)