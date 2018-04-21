import os
import pandas as pd
import sqlalchemy


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
    'state_window'
    ]

complete_data = pd.DataFrame(columns=cols)

# iterate over all csv and bind them together
for dirname, dirnames, filenames in os.walk('~/Room-Climate-Datasets'):
    for subdirname in dirnames:
        print(os.path.join(dirname, subdirname))