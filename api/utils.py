import datetime
import json
import pandas as pd
from core.composer.gp_composer.gp_composer import GPComposer, GPComposerRequirements
from core.composer.visualisation import ComposerVisualiser
from core.models.data import InputData, OutputData
from core.repository.model_types_repository import (
    ModelTypesRepository
)
from core.repository.quality_metrics_repository import ClassificationMetricsEnum, RegressionMetricsEnum, \
    MetricsRepository
from core.repository.tasks import Task, TaskTypesEnum


def compose_fedot_model(train_file_path: str,
                        task: Task,
                        learning_time: int,
                        is_visualise=False):
    # the choice of the metric for the chain quality assessment during composition
    if task == Task(TaskTypesEnum.classification):
        metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC_penalty)
    elif task == Task(TaskTypesEnum.regression):
        metric_function = MetricsRepository().metric_by_id(RegressionMetricsEnum.RMSE_penalty)
    learning_time = datetime.timedelta(minutes=learning_time)

    dataset_to_compose = InputData.from_csv(train_file_path, task=task)

    # the search of the models provided by the framework that can be used as nodes in a chain for the selected task
    available_model_types, _ = ModelTypesRepository().suitable_model(task_type=task.task_type)

    # the choice and initialisation of the GP search
    composer_requirements = GPComposerRequirements(
        primary=available_model_types,
        secondary=available_model_types, max_arity=3,
        max_depth=3, pop_size=20, num_of_generations=20,
        crossover_prob=0.8, mutation_prob=0.8, max_lead_time=learning_time)

    # Create GP-based composer
    composer = GPComposer()

    # the optimal chain generation by composition - the most time-consuming task
    chain_evo_composed = composer.compose_chain(data=dataset_to_compose,
                                                initial_chain=None,
                                                composer_requirements=composer_requirements,
                                                metrics=metric_function,
                                                is_visualise=False)

    chain_evo_composed.fit(input_data=dataset_to_compose, verbose=True)

    if is_visualise:
        ComposerVisualiser.visualise(chain_evo_composed)

    return chain_evo_composed


def load_model(fedot_model_path: str):
    with open(fedot_model_path, 'r') as json_file:
        fitted_model = json.load(json_file)
    return fitted_model


def save_predict(predicted_data: OutputData):
    return pd.DataFrame({'Index': predicted_data.idx,
                         'Prediction': predicted_data.predict}).to_csv(r'./predictions.csv', index=False)
