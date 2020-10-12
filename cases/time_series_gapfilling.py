import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error

from core.utils import project_root
from utilities.ts_gapfilling import AdvancedGapfiller


def check_metrics(data):
    gap_array = np.array(data['With_gap'])
    gap_ids = np.argwhere(gap_array == -100.0)

    actual = np.array(data['Temperature'])[gap_ids]
    ridge_predicted = np.array(data['Ridge'])[gap_ids]
    composite_predicted = np.array(data['Composite'])[gap_ids]

    for i, predicted in enumerate([ridge_predicted, composite_predicted]):
        if i == 0:
            print('Inverted ridge regression model')
        else:
            print('Composite chain of 5 models')

        mae_metric = mean_absolute_error(actual, predicted)
        print('Mean absolute error -', round(mae_metric, 2))

        rmse_metric = (mean_squared_error(actual, predicted)) ** 0.5
        print('RMSE -', round(rmse_metric, 2))

        median_ae_metric = median_absolute_error(actual, predicted)
        print('Median absolute error -', round(median_ae_metric, 2), '\n')


def plot_result(data):
    # Plot predicted values
    gap_array = np.array(data['With_gap'])
    masked_array = np.ma.masked_where(gap_array == -100.0, gap_array)

    plt.plot(data['Date'], data['Temperature'], c='blue', alpha=0.5, label='Actual values', linewidth=1)
    plt.plot(data['Date'], data['Ridge'], c='orange', alpha=0.8, label='Inverse ridge gapfilling', linewidth=1)
    plt.plot(data['Date'], data['Composite'], c='red', alpha=0.8, label='Composite gapfilling', linewidth=1)
    plt.plot(data['Date'], masked_array, c='blue')
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
    dataframe['Date'] = pd.to_datetime(dataframe['Date'])

    # Filling in gaps based on inverted ridge regression model
    Gapfiller_ridge = AdvancedGapfiller(gap_value=-100.0)
    withoutgap_arr_ridge = Gapfiller_ridge.inverse_ridge(np.array(dataframe['With_gap']), max_window_size=250)
    dataframe['Ridge'] = withoutgap_arr_ridge

    # Filling in gaps based on a chain of 5 models
    Gapfiller_composite = AdvancedGapfiller(gap_value=-100.0)
    withoutgap_arr_composite = Gapfiller_composite.composite_fill_gaps(np.array(dataframe['With_gap']),
                                                                       max_window_size=1000)
    dataframe['Composite'] = withoutgap_arr_composite

    # Display metrics
    check_metrics(dataframe)

    # Visualise predictions
    plot_result(dataframe)
