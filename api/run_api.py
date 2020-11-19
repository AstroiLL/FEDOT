from dataclasses import dataclass
from sklearn.metrics import roc_auc_score as roc_auc
from sklearn.metrics import mean_squared_error as mse
from api.utils import run_ml_problem, load_model, save_predict
from core.models.data import InputData
from core.repository.tasks import Task, TaskTypesEnum


@dataclass
class FedotModel:
    train_file_path: str
    test_file_path: str
    ml_task: str
    learning_time: int = 10
    case: str = 'Fedot_model'
    fit_flag: bool = False

    def _get_params(self):
        task_dict = {'Regression': Task(TaskTypesEnum.regression),
                     'Classification': Task(TaskTypesEnum.classification),
                     }
        self.ml_task = task_dict[self.ml_task]
        param_dict = {'train_file_path': self.train_file_path,
                      'task': self.ml_task,
                      'learning_time': self.learning_time,
                      }
        return param_dict

    def _get_model(self):
        execution_params = self._get_params()
        if self.fit_flag:
            model = run_ml_problem(**execution_params)
        else:
            model = load_model(**execution_params)
        return model

    def fit(self):
        self.fit_flag = True
        return self._get_model()

    def predict(self):
        current_model = self.fit()
        self.test_data = InputData.from_csv(self.test_file_path, task=self.ml_task)
        self.predicted = current_model.predict(self.test_data)
        save_predict(self.predicted)
        #current_model.save_chain("./fitted_chain_convert_to_json.json")
        return self.predicted.predict

    def quality_metric(self):
        if self.ml_task == Task(TaskTypesEnum.regression):
            metric_value = mse(y_true=self.test_data.target, y_pred=self.predicted.predict, squared=False)
        elif self.ml_task == Task(TaskTypesEnum.classification):
            metric_value = round(roc_auc(y_true=self.test_data.target,
                                         y_score=self.predicted.predict), 3)
        else:
            print('Current version doesnt support your type of ML task')
            metric_value = 0
        return metric_value
