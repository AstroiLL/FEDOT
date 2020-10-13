import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error

from core.utils import project_root
from utilities.ts_gapfilling import AdvancedGapfiller


def print_metrics(data):
    gap_array = np.array(data['with_gap'])
    gap_ids = np.argwhere(gap_array == -100.0)

    actual = np.array(data['temperature'])[gap_ids]
    ridge_predicted = np.array(data['ridge'])[gap_ids]
    composite_predicted = np.array(data['composite'])[gap_ids]

    for i, predicted in enumerate([ridge_predicted, composite_predicted]):
        if i == 0:
            print('Inverted ridge regression model')
        else:
            print('Composite chain of 5 models')

        mae_metric = mean_absolute_error(actual, predicted)
        print(f"Mean absolute error -, {round(mae_metric, 2)}")

        rmse_metric = (mean_squared_error(actual, predicted)) ** 0.5
        print(f"Root mean squared error -, {round(rmse_metric, 2)}")

        median_ae_metric = median_absolute_error(actual, predicted)
        print(f"Median absolute error -, {round(median_ae_metric, 2)} \n")


def plot_result(data):
    # Plot predicted values
    gap_array = np.array(data['with_gap'])
    masked_array = np.ma.masked_where(gap_array == -100.0, gap_array)

    plt.plot(data['date'], data['temperature'], c='blue', alpha=0.5, label='Actual values', linewidth=1)
    plt.plot(data['date'], data['ridge'], c='orange', alpha=0.8, label='Inverse ridge gapfilling', linewidth=1)
    plt.plot(data['date'], data['composite'], c='red', alpha=0.8, label='Composite gapfilling', linewidth=1)
    plt.plot(data['date'], masked_array, c='blue')
    plt.grid()
    plt.legend()
    plt.show()


# Example of using the algorithm to fill in gaps in a time series with a gap of 1000 elements
# The data is daily air temperature values from the weather station
if __name__ == '__main__':
    # Load dataframe
    file_path = 'cases/data/gapfilling/TS_temperature_gapfilling.csv'
    full_path = os.path.join(str(project_root()), file_path)
    dataframe = pd.read_csv(full_path)
    dataframe['date'] = pd.to_datetime(dataframe['date'])

    # Filling in gaps based on inverted ridge regression model
    gapfiller_ridge = AdvancedGapfiller(gap_value=-100.0)
    without_gap_arr_ridge = gapfiller_ridge.inverse_ridge(np.array(dataframe['with_gap']), max_window_size=250)
    dataframe['ridge'] = without_gap_arr_ridge

    # Filling in gaps based on a chain of 5 models
    gapfiller_composite = AdvancedGapfiller(gap_value=-100.0)
    without_gap_arr_composite = gapfiller_composite.composite_fill_gaps(np.array(dataframe['with_gap']),
                                                                        max_window_size=1000)
    dataframe['composite'] = without_gap_arr_composite

    # Display metrics
    print_metrics(dataframe)

    # Visualise predictions
    plot_result(dataframe)
