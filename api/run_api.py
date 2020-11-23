import numpy as np
from api.utils import compose_fedot_model, save_predict, array_to_input_data
from fedot.core.models.data import InputData
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.composer.chain import Chain
from fedot.core.composer.metrics import F1Metric, MaeMetric, RmseMetric, RocAucMetric


def default_evo_params():
    return {'max_depth': 3,
            'max_arity': 3,
            'pop_size': 20,
            'num_of_generations': 20,
            'learning_time': 5}


def fedot_runner(ml_task: str,
                 composer_params: dict = None,
                 fedot_model_path: str = './fedot_model.json',
                 metric_name: str = 'RMSE'):
    fedot_model = Fedot(ml_task=ml_task,
                        composer_params=default_evo_params(),
                        fedot_model_path=fedot_model_path,
                        metric_name=metric_name)
    if composer_params is not None:
        fedot_model = Fedot(ml_task=ml_task,
                            composer_params=composer_params,
                            fedot_model_path=fedot_model_path,
                            metric_name=metric_name)

    return fedot_model


class Fedot(object):

    def __init__(self,
                 ml_task: str,
                 composer_params: dict,
                 fedot_model_path: str,
                 metric_name: str,
                 current_model: Chain = None):
        self.ml_task = ml_task
        self.composer_params = composer_params
        self.fedot_model_path = fedot_model_path
        self.metric_name = metric_name
        self.current_model = current_model
        task_dict = {'reg': Task(TaskTypesEnum.regression),
                     'clf': Task(TaskTypesEnum.classification),
                     'cluster': Task(TaskTypesEnum.clustering)
                     }
        basic_metric_dict = {'reg': 'RMSE',
                             'clf': 'ROCAUC',
                             }

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
            features: np.array = None,
            target: np.array = None,
            csv_path: str = None):
        if csv_path is None:
            self.train_data = array_to_input_data(features_array=features,
                                                  target_array=target,
                                                  task_type=self.ml_task)
        else:
            self.train_data = InputData.from_csv(csv_path, task=self.ml_task)
        return self._get_model()

    def predict(self,
                features: np.array = None,
                target: np.array = None,
                csv_path: str = None,
                save_model: bool = False):
        if self.current_model is None:
            self.current_model = self._get_model()

        if csv_path is None:
            self.test_data = array_to_input_data(features_array=features,
                                                 target_array=target,
                                                 task_type=self.ml_task)
        else:
            self.test_data = InputData.from_csv(csv_path, task=self.ml_task)

        self.predicted = self.current_model.predict(self.test_data)
        save_predict(self.predicted)
        if save_model:
            self.current_model.save_chain(self.fedot_model_path)
        return self.predicted.predict

    def quality_metric(self, metric_name: str = None):
        if metric_name is None:
            metric_name = self.metric_name

        __metric_dict = {'RMSE': RmseMetric.metric,
                         'MAE': MaeMetric.metric,
                         'ROCAUC': RocAucMetric.metric,
                         'F1': F1Metric.metric,
                         }

        metric_value = round(__metric_dict[metric_name](reference=self.test_data,
                                                        predicted=self.predicted), 3)
        return metric_value
