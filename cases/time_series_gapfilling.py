import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error

from core.utils import project_root
from utilities.ts_gapfilling import ModelGapFiller


def print_metrics(dataframe):
    """
    The function displays 3 metrics: Mean absolute error,
    Root mean squared error and Median absolute error

    :param: dataframe with columns 'date','temperature','ridge','composite',
    'with_gap'
    """

    gap_array = np.array(dataframe['with_gap'])
    gap_ids = np.argwhere(gap_array == -100.0)

    actual = np.array(dataframe['temperature'])[gap_ids]
    ridge_predicted = np.array(dataframe['ridge'])[gap_ids]
    composite_predicted = np.array(dataframe['composite'])[gap_ids]

    model_labels = ['Inverted ridge regression', 'Composite model']
    for predicted, model_label in zip(
            [ridge_predicted, composite_predicted], model_labels):
        print(f"{model_label}")

        mae_metric = mean_absolute_error(actual, predicted)
        print(f"Mean absolute error - {mae_metric:.2f}")

        rmse_metric = (mean_squared_error(actual, predicted)) ** 0.5
        print(f"Root mean squared error - {rmse_metric:.2f}")

        median_ae_metric = median_absolute_error(actual, predicted)
        print(f"Median absolute error - {median_ae_metric:.2f} \n")


def plot_result(dataframe):
    """
    The function draws a graph based on the dataframe

    :param: dataframe with columns 'date','temperature','ridge','composite',
    'with_gap'
    """

    gap_array = np.array(dataframe['with_gap'])
    masked_array = np.ma.masked_where(gap_array == -100.0, gap_array)

    plt.plot(dataframe['date'], dataframe['temperature'], c='blue',
             alpha=0.5, label='Actual values', linewidth=1)
    plt.plot(dataframe['date'], dataframe['ridge'], c='orange',
             alpha=0.8, label='Inverse ridge gapfilling', linewidth=1)
    plt.plot(dataframe['date'], dataframe['composite'], c='red',
             alpha=0.8, label='Composite gapfilling', linewidth=1)
    plt.plot(dataframe['date'], masked_array, c='blue')
    plt.grid()
    plt.legend()
    plt.show()


# Example of using the algorithm to fill in gaps in a time series with a gap
# of 1000 elements
# The data is daily air temperature values from the weather station
if __name__ == '__main__':
    # Load dataframe
    file_path = 'cases/data/gapfilling/TS_temperature_gapfilling.csv'
    full_path = os.path.join(str(project_root()), file_path)
    dataframe = pd.read_csv(full_path)
    dataframe['date'] = pd.to_datetime(dataframe['date'])

    # Filling in gaps based on inverted ridge regression model
    fedot_gapfiller = ModelGapFiller(gap_value=-100.0)
    without_gap_arr_ridge = \
        fedot_gapfiller.inverse_ridge(np.array(dataframe['with_gap']),
                                      max_window_size=250)
    dataframe['ridge'] = without_gap_arr_ridge

    # Filling in gaps based on a chain of 5 models
    without_gap_arr_composite = \
        fedot_gapfiller.composite_model(np.array(dataframe['with_gap']),
                                        max_window_size=1000)
    dataframe['composite'] = without_gap_arr_composite

    # Display metrics
    print_metrics(dataframe)

    # Visualise predictions
    plot_result(dataframe)
