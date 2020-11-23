import numpy as np
from api.utils import compose_fedot_model, save_predict, array_to_input_data
from fedot.core.models.data import InputData
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.composer.metrics import F1Metric, MaeMetric, RmseMetric, RocAucMetric


def default_evo_params():
    return {'max_depth': 3,
            'max_arity': 3,
            'pop_size': 20,
            'num_of_generations': 20,
            'learning_time': 2}


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
            features,
            target='target'):
        if type(features) == np.ndarray:
            self.train_data = array_to_input_data(features_array=features,
                                                  target_array=target,
                                                  task_type=self.ml_task)
        elif type(features) == str:
            self.train_data = InputData.from_csv(features, task=self.ml_task, target_column=target)
        else:
            print('Please specify a features as path to csv file or as Numpy array')
        return self._get_model()

    def predict(self,
                features,
                target='target'):
        if self.current_model is None:
            self.current_model = self._get_model()

        if type(features) == np.ndarray:
            self.test_data = array_to_input_data(features_array=features,
                                                 target_array=target,
                                                 task_type=self.ml_task)
        elif type(features) == str:
            self.test_data = InputData.from_csv(features, task=self.ml_task, target_column=target)
        else:
            print('Please specify a features as path to csv file or as Numpy array')

        self.predicted = self.current_model.predict(self.test_data)
        save_predict(self.predicted)
        return self.predicted.predict

    def save_model(self):
        return self.current_model.save_chain(self.fedot_model_path)

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
