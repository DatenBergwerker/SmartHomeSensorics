import re
import os
import pandas as pd
import yaml

PATH, DATA_PATH =

# setup container
cols = [
    'entry_id',
    'absolute_timestamp',
    'relative_timestamp',
    'node_id',
    'temperature',
    'relative_humidity',
    'light_sensor_1_wvl_nm',
    'light_sensor_2_wvl_nm',
    'no_occupants',
    'occupant_activity',
    'state_door',
    'state_window',
    'room_location',
    'measurement_no'
    ]

complete_data = pd.DataFrame(columns=cols)

room_extractor = re.compile(r'datasets-location_([ABC])$')
numeration_extractor = re.compile(r'-measurement([0-9]{2}).csv$')

# iterate over all csv and bind them together
for dirname, dirnames, filenames in os.walk(DATA_PATH):

    try:
        name = room_extractor.search(string=dirname).group(1)
    except AttributeError as e:
        name = None
        continue

    if name:
        for file in filenames:
            counter = numeration_extractor.search(string=file).group(1)
            full_path = os.path.join(dirname, file)
            print(full_path)

            temp_df = pd.read_csv(full_path, header=None)
            temp_df['room_location'] = name
            temp_df['measurement_no'] = counter
            temp_df.columns = cols
            complete_data = complete_data.append(
                temp_df
            )

complete_sorted = complete_data.sort_values(by=['room_location', 'measurement_no', 'entry_id'])
complete_sorted.to_csv('SmartHomeSensorics_full_data.csv', header=True, index=False)