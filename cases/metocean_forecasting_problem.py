import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse

from core.composer.chain import Chain
from core.composer.node import PrimaryNode, SecondaryNode
from core.models.data import InputData, OutputData
from core.repository.dataset_types import DataTypesEnum
from core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from core.utils import project_root
from copy import deepcopy


def get_composite_chain():
    chain = Chain()
    node_trend = PrimaryNode('trend_data_model')
    node_lstm_trend = SecondaryNode('linear', nodes_from=[node_trend])

    node_residual = PrimaryNode('residual_data_model')
    node_ridge_residual = SecondaryNode('rfr', nodes_from=[node_residual])

    node_final = SecondaryNode('additive_data_model',
                               nodes_from=[node_ridge_residual, node_lstm_trend])
    chain.add_node(node_final)
    return chain


def calculate_validation_metric(pred: OutputData, valid: InputData,
                                name: str, predict_start=0,
                                target_start=0, is_visualise=False) -> float:
    forecast_length = valid.task.task_params.forecast_length

    # skip initial part of time series
    predicted = pred.predict[predict_start:]
    real = valid.target[target_start:]

    # plot results
    if is_visualise:
        compare_plot(predicted, real,
                     forecast_length=forecast_length,
                     model_name=name)

    # the quality assessment for the simulation results
    slices = []
    for i in range(forecast_length):
        if i + 1 == forecast_length:
            slices.append(real[i:])
        else:
            slices.append(real[i : i+1-forecast_length])

    rmse = mse(y_true=np.column_stack(slices),
               y_pred=predicted,
               squared=False)

    return rmse


def compare_plot(predicted, real, forecast_length, model_name):
    plt.clf()
    _, ax = plt.subplots()
    plt.plot(real, linewidth=1, label="Observed", alpha=0.4)
    plt.plot(predicted, linewidth=1, label="Predicted", alpha=0.6)
    ax.legend()
    plt.xlabel('Time, h')
    plt.ylabel('SSH, cm')
    plt.title(f'Sea surface height forecast for {forecast_length} hours with {model_name}')
    plt.show()

def run_metocean_forecasting_problem(train_file_path, test_file_path,
                                     forecast_length=100, max_window_size=100,
                                     period=1, is_visualise=False):
    # specify the task to solve
    task_to_solve = Task(TaskTypesEnum.ts_forecasting,
                         TsForecastingParams(forecast_length=forecast_length,
                                             max_window_size=max_window_size,
                                             period=period))

    full_path_train = os.path.join(str(project_root()), train_file_path)
    dataset_to_train = InputData.from_csv(
        full_path_train, task=task_to_solve, data_type=DataTypesEnum.ts)

    # n = 100
    # dataset_to_train.features = dataset_to_train.features[:n]
    # dataset_to_train.target = dataset_to_train.target[:n]
    # dataset_to_train.idx = dataset_to_train.idx[:n]

    # a dataset for a final validation of the composed model
    full_path_test = os.path.join(str(project_root()), test_file_path)
    dataset_to_validate = InputData.from_csv(
        full_path_test, task=task_to_solve, data_type=DataTypesEnum.ts)

    chain_simple = Chain()
    node_simple = PrimaryNode('lasso')
    chain_simple.add_node(node_simple)

    chain_simple.fit(input_data=dataset_to_train, verbose=False)
    rmse_on_valid_simple = calculate_validation_metric(
        chain_simple.predict(dataset_to_validate), dataset_to_validate,
        f'full-simple_{forecast_length}', 
        0,
        max_window_size,
        is_visualise)
    print(f'RMSE simple: {rmse_on_valid_simple}')

    chain_lstm = get_composite_chain()
    chain_lstm.fit(input_data=dataset_to_train, verbose=False)
    rmse_on_valid_lstm_only = calculate_validation_metric(
        chain_lstm.predict(dataset_to_validate), dataset_to_validate,
        f'full-lstm-only_{forecast_length}', 
        max_window_size, 
        max_window_size-forecast_length+1,
        is_visualise)
    print(f'RMSE LSTM composite: {rmse_on_valid_lstm_only}')

    return rmse_on_valid_simple


if __name__ == '__main__':
    # the dataset was obtained from NEMO model simulation for sea surface height

    # a dataset that will be used as a train and test set during composition
    file_path_train = 'cases/data/metocean/metocean_data_train.csv'
    full_path_train = os.path.join(str(project_root()), file_path_train)

    # a dataset for a final validation of the composed model
    file_path_test = 'cases/data/metocean/metocean_data_test.csv'
    full_path_test = os.path.join(str(project_root()), file_path_test)

    run_metocean_forecasting_problem(full_path_train, full_path_test,
                                     forecast_length=100, max_window_size=100, is_visualise=True)
