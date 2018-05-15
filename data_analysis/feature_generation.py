import os
from multiprocessing import cpu_count
import pandas as pd
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import roll_time_series, impute
from data.data_loader import DataLoader

# Settings

# relevant features for tsfresh generation
FEATURES = [
    'node_id',
    'entry_id',
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
    'relative_timestamp',
    'absolute_timestamp',
    'binary_target',
    'no_occupants',
    'occupant_activity'
]

# remove columns with more than 10 % missing values based on the shared room data (more robust) or roomwise data (more tuned)
SELECTION = 'roomwise'

# size of windows in number of past observations (≈ 4 sec, ± 1 sec between observations)
WINDOW_SIZE = 30

# number of cores used for the feature computation
NCORES = cpu_count() - 1

# number of measurement csvs per room
room_params = {'A': 60,
               'B': 44,
               'C': 40}

loader = DataLoader(path=os.path.join(os.getcwd(), 'SmartHomeSensorics_full_data.csv'))

feature_matrix_list = []

for room in room_params:
    for measurement in range(1, room_params[room] + 1):
        print(f'Room: {room}, Measurement: {measurement}')
        subset_data = loader.return_experiment_measurement(room_location=room, measurement_no=measurement,
                                                           sensor_node=1)
        subset_data.set_index('entry_id', inplace=True)
        subset_data['entry_id'] = subset_data.index
        target_vector = subset_data['binary_target']
        subset_data.sort_values(['node_id', 'entry_id'], inplace=True)
        ts_for_rolling = subset_data[FEATURES]
        ts_for_rolling = roll_time_series(ts_for_rolling, column_id='node_id', column_sort='entry_id',
                                          column_kind=None, rolling_direction=1, max_timeshift=WINDOW_SIZE)
        temp_df = extract_features(ts_for_rolling, n_jobs=NCORES, column_sort='entry_id',
                                   column_id='node_id', column_kind=None, show_warnings=False)
        impute_df = impute(temp_df)
        temp_sel_df = select_features(X=impute_df, y=target_vector, n_jobs=NCORES)
        join_df = subset_data[FEATURES + JOIN_FEATURES]
        join_df = join_df.join(temp_sel_df)
        feature_matrix_list.append(join_df)

full_feature_matrix = pd.concat(feature_matrix_list, axis=0, ignore_index=True)
full_feature_matrix = full_feature_matrix.reset_index(drop=False)
full_feature_matrix.to_csv('SmartHomeSensorics_selected_feature_matrix.csv', index=False)

full_feature_matrix = pd.read_csv('SmartHomeSensorics_selected_feature_matrix.csv')

# throw out columns with more than 15 % missing

if SELECTION == 'all_rooms':
    na_cols = [col for col in full_feature_matrix.columns
               if full_feature_matrix[col].isnull().sum() / full_feature_matrix.shape[0] > 0.15]
    full_feature_matrix = full_feature_matrix.drop(na_cols, axis=1)
    full_feature_matrix['rowmiss'] = full_feature_matrix.isnull().sum(axis=1)
    full_feature_matrix_cleaned_na = full_feature_matrix.loc[full_feature_matrix.rowmiss < 30]
    full_feature_matrix_cleaned_na = full_feature_matrix_cleaned_na.drop('rowmiss', axis=1)
    full_feature_matrix_cleaned_na.to_csv('SmartHomeSensorics_selected_feature_matrix_cleaned.csv', index=False)

elif SELECTION == 'roomwise':
    for room in room_params:
        print(f'Processing room {room}')
        roomwise_feature_matrix = full_feature_matrix.loc[full_feature_matrix.room_location == room]

        na_cols = [col for col in full_feature_matrix.columns
                   if full_feature_matrix[col].isnull().sum() / full_feature_matrix.shape[0] > 0.15]

        roomwise_feature_matrix = roomwise_feature_matrix.drop(na_cols, axis=1)
        roomwise_feature_matrix['rowmiss'] = roomwise_feature_matrix.isnull().sum(axis=1)
        roomwise_feature_matrix_cleaned_na = roomwise_feature_matrix.loc[roomwise_feature_matrix.rowmiss < 30]
        roomwise_feature_matrix_cleaned_na = roomwise_feature_matrix_cleaned_na.drop('rowmiss', axis=1)
        roomwise_feature_matrix_cleaned_na.to_csv(f'SmartHomeSensorics_selected_feature_matrix_cleaned_room_{room}.csv',
                                                  index=False)