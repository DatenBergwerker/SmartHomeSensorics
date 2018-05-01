import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from data.data_loader import DataLoader


def make_measure_pairplot(subset_data: pd.DataFrame,
                          plot_params: dict,
                          hue: bool = True) -> plt.figure:
    """
    Draw the pairplot for the specified measurement subset.

    :param hue: A boolean to specify if the plot is to be drawn with binary target display.
    :param plot_params: A dictionary containing plot parameters: columns to plot and plot title.
    :param subset_data: The subsetted DataFrame
    :return: The finished figure object containing the formatted pairplot
    """
    if hue:
        figure = sns.pairplot(subset_data[plot_params['cols']], hue='binary_target')
    else:
        figure = sns.pairplot(subset_data[plot_params['cols']])

    if plot_params['sensor']:
        figure.fig.set_title(f'Feature Pairwise Plot for Room {plot_params["room"]}, '
                             f'Measurement {plot_params["measurement"]}, Sensor Node {plot_params["sensor"]}')
    else:
        figure.fig.set_title(f'Feature Pairwise Plot for Room {plot_params["room"]}, '
                             f'Measurement {plot_params["measurement"]}')
    return figure


if __name__ == '__main__':
    loader = DataLoader('SmartHomeSensorics_full_data.csv')

    full_dataset = loader.raw_data
    plot_cols = full_dataset.drop(['entry_id', 'absolute_timestamp',
                                   'relative_timestamp', 'measurement_no'], axis=1).columns

    plot_params = {
        'cols': plot_cols,
        'sensor': None
    }
    for room in full_dataset['room_location'].unique().tolist():
        measurements = full_dataset.loc[full_dataset.room_location == room, 'measurement_no'].max()

        for measurement in range(measurements):
            print(f'pairplot_{room}_{measurement}_sensor_{plot_params["sensor"]}.png')
            subset_data = loader.return_experiment_measurement(room_location=room,
                                                               measurement_no=measurement + 1,
                                                               sensor_node=None)
            plot_params.update({
                'room': room,
                'measurement': measurement
            })
            figure = make_measure_pairplot(subset_data=subset_data, plot_params=plot_params, hue=True)
            figure.savefig(os.path.join('data_analysis',
                                        'plots' f'pairplot_{room}_{measurement}_sensor_{plot_params["sensor"]}.png'))
            plt.clf()
