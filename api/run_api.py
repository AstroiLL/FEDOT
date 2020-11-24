from typing import Union
import numpy as np
from api.utils import compose_fedot_model, save_predict, array_to_input_data
from fedot.core.models.data import InputData
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.composer.metrics import F1Metric, MaeMetric, RmseMetric, RocAucMetric


def default_evo_params(max_depth: int = 3,
                       max_arity: int = 3,
                       pop_size: int = 20,
                       num_of_generations: int = 20,
                       learning_time: int = 2):
    return {'max_depth': max_depth,
            'max_arity': max_arity,
            'pop_size': pop_size,
            'num_of_generations': num_of_generations,
            'learning_time': learning_time}


def check_data_type(ml_task: Task,
                    features: Union[str, np.ndarray],
                    target: Union[str, np.ndarray] = None):
    if type(features) == np.ndarray:
        if target is None:
            target = np.array([])

        data = array_to_input_data(features_array=features,
                                   target_array=target,
                                   task_type=ml_task)
    elif type(features) == str:
        if target is None:
            target = 'target'

        data = InputData.from_csv(features, task=ml_task, target_column=target)
    else:
        print('Please specify a features as path to csv file or as Numpy array')
    return data


class Fedot(object):

    def __init__(self,
                 ml_task: str,
                 composer_params: dict = None,
                 fedot_model_path: str = './fedot_model.json'):
        self.ml_task = ml_task
        self.fedot_model_path = fedot_model_path
        self.composer_params = composer_params

        if self.composer_params is None:
            self.composer_params = default_evo_params()
        else:
            self.composer_params = {**default_evo_params(), **self.composer_params}

        task_dict = {'reg': Task(TaskTypesEnum.regression),
                     'clf': Task(TaskTypesEnum.classification),
                     'cluster': Task(TaskTypesEnum.clustering),
                     'time_series': Task(TaskTypesEnum.ts_forecasting)
                     }
        basic_metric_dict = {'reg': 'RMSE',
                             'clf': 'ROCAUC',
                             'cluster': 'Silhouette',
                             'time_series': 'RMSE'
                             }

        if self.ml_task == 'cluster' or self.ml_task == 'time_series':
            print('This type of task is not not supported in API now')

        self.metric_name = basic_metric_dict[self.ml_task]
        self.ml_task = task_dict[self.ml_task]

    def _get_params(self):
        param_dict = {'train_data': self.train_data,
                      'task': self.ml_task,
                      }
        return {**param_dict, **self.composer_params}

    def _get_model(self):
        execution_params = self._get_params()
        self.current_model = compose_fedot_model(**execution_params)
        return self.current_model

    def fit(self,
            features: Union[str, np.ndarray],
            target: Union[str, np.ndarray] = 'target'):
        self.train_data = check_data_type(ml_task=self.ml_task,
                                          features=features,
                                          target=target)
        return self._get_model()

    def predict(self,
                features: Union[str, np.ndarray]):
        if self.current_model is None:
            self.current_model = self._get_model()

        self.test_data = check_data_type(ml_task=self.ml_task,
                                         features=features)

        self.predicted = self.current_model.predict(self.test_data)
        save_predict(self.predicted)
        return self.predicted.predict

    def save_model(self):
        return self.current_model.save_chain(self.fedot_model_path)

    def quality_metric(self,
                       target: np.ndarray = None,
                       metric_name: str = None):
        if metric_name is None:
            metric_name = self.metric_name

        if target is not None:
            self.test_data.target = target

        __metric_dict = {'RMSE': RmseMetric.metric,
                         'MAE': MaeMetric.metric,
                         'ROCAUC': RocAucMetric.metric,
                         'F1': F1Metric.metric,
                         }

        metric_value = round(__metric_dict[metric_name](reference=self.test_data,
                                                        predicted=self.predicted), 3)
        return metric_value
