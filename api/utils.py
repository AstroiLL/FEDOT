import datetime
import json
import pandas as pd
import numpy as np
from fedot.core.composer.gp_composer.gp_composer import GPComposerBuilder, GPComposerRequirements
from fedot.core.models.data import InputData, OutputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.model_types_repository import (
    ModelTypesRepository
)
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum, RegressionMetricsEnum, \
    MetricsRepository
from fedot.core.repository.tasks import Task, TaskTypesEnum


def load_model(fedot_model_path: str):
    with open(fedot_model_path, 'r') as json_file:
        fitted_model = json.load(json_file)
    return fitted_model


def save_predict(predicted_data: OutputData):
    return pd.DataFrame({'Index': predicted_data.idx,
                         'Prediction': predicted_data.predict}).to_csv(r'./predictions.csv', index=False)


def array_to_input_data(features_array: np.array,
                        target_array: np.array,
                        task_type: Task = Task(TaskTypesEnum.classification)):
    data_type = DataTypesEnum.table
    idx = np.arange(len(features_array))

    return InputData(idx=idx, features=features_array, target=target_array, task=task_type, data_type=data_type)


def compose_fedot_model(train_data: InputData,
                        task: Task,
                        max_depth: int,
                        max_arity: int,
                        pop_size: int,
                        num_of_generations: int,
                        learning_time: int = 5,
                        ):
    # the choice of the metric for the chain quality assessment during composition
    if task == Task(TaskTypesEnum.classification):
        metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC_penalty)
    elif task == Task(TaskTypesEnum.regression):
        metric_function = MetricsRepository().metric_by_id(RegressionMetricsEnum.RMSE_penalty)
    learning_time = datetime.timedelta(minutes=learning_time)

    # the search of the models provided by the framework that can be used as nodes in a chain for the selected task
    available_model_types, _ = ModelTypesRepository().suitable_model(task_type=task.task_type)

    # the choice and initialisation of the GP search
    composer_requirements = GPComposerRequirements(
        primary=available_model_types,
        secondary=available_model_types, max_arity=max_arity,
        max_depth=max_depth, pop_size=pop_size, num_of_generations=num_of_generations,
        crossover_prob=0.8, mutation_prob=0.8, max_lead_time=learning_time)

    # Create GP-based composer
    builder = GPComposerBuilder(task).with_requirements(composer_requirements).with_metrics(metric_function)
    gp_composer = builder.build()

    chain_gp_composed = gp_composer.compose_chain(data=train_data)

    chain_gp_composed.fit_from_scratch(input_data=train_data)

    return chain_gp_composed
