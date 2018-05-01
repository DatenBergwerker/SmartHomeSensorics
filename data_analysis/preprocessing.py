# TODO: Build timeseries features (tsfresh?), otherwise sma, treshold values, min to max diff etc
# TODO: Normalize data
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tsfresh import extract_relevant_features

from data.data_loader import DataLoader

def scale_dataset(subset_data: pd.DataFrame, cols: dict):
    """
    Scale all numeric columns in the dataset with the chosen scaler.

    :param subset_data:
    :return:
    """


def feature_generation(subset_data: pd.DataFrame, cols: list, window: int):
    """
    Function to generate the features for each timeseries and merge them all together.
    The feature extracting is based on the 'FeatuRe Extraction based on Scalable Hypothesis tests'.

    :param subset_data:
    :param cols:
    :param window:
    :return:
    """

subset_data =