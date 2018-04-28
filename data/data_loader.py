import pandas as pd


class DataLoader:
    """
    Base interface to interact with the raw data.
    """

    def __init__(self, path: str):
        self.raw_data = pd.read_csv(path)
        self._set_data_types()
        self._build_binary_target()

    def _set_data_types(self):
        """
        This functions sets the correct data types for all columns after csv import.
        """
        temp_df = self.raw_data
        cols = temp_df.drop('room_location', axis=1).columns
        temp_df[cols] = temp_df[cols].apply(pd.to_numeric)
        temp_df['room_location'] = temp_df['room_location'].astype(str)
        self.raw_data = temp_df

    def _build_binary_target(self):
        """
        This functions just builds a binary representation if someone is currently in the room.
        """
        self.raw_data['binary_target'] = 0
        self.raw_data.loc[self.raw_data.no_occupants > 0, 'binary_target'] = 1

    def return_experiment_measurement(self,
                                      room_location: str,
                                      measurement_no: int,
                                      sensor_node: int = None) -> pd.DataFrame:
        """
        This function returns a single subset of a measurement. It can return all
        sensor data or just from a special node.
        """
        if not sensor_node:
            return self.raw_data.loc[(self.raw_data.room_location == room_location) &
                                     (self.raw_data.measurement_no == measurement_no)]
        else:
            return self.raw_data.loc[(self.raw_data.room_location == room_location) &
                                     (self.raw_data.measurement_no == measurement_no) &
                                     (self.raw_data.node_id == sensor_node)]


        # IDEA: Moving cross validation window
        # IDEA: Generate train / test splits for each dataset