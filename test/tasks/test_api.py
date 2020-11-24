from api.run_api import Fedot
import pandas as pd
import os
import numpy as np
from fedot.core.utils import project_root
from sklearn.model_selection import train_test_split

task_type = 'reg'
composer_params = {'max_depth': 3,
                   'learning_time': 1}


def get_api_data_paths():
    file_path_train = 'cases/data/oil_chemistry/train.csv'
    file_path_test = 'cases/data/oil_chemistry/test_1.csv'
    full_path_train = os.path.join(str(project_root()), file_path_train)
    full_path_test = os.path.join(str(project_root()), file_path_test)

    return full_path_train, full_path_test


def get_regression_data():
    train_full, test = get_api_data_paths()
    train_file = pd.read_csv(train_full)
    x, y = train_file.loc[:, ~train_file.columns.isin(['target'])].values, train_file['target'].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=24)
    return x_train, x_test, y_train, y_test


def test_api_baseline():
    x_train, x_test, y_train, y_test = get_regression_data()
    model_baseline = Fedot(ml_task=task_type)

    model_baseline.fit(features=x_train,
                       target=y_train)
    model_baseline.predict(features=x_test)

    metric_baseline = model_baseline.quality_metric(target=y_test)
    threshold = np.std(y_test)

    assert metric_baseline < threshold


def test_api_advanced():
    train, test = get_api_data_paths()
    model_advanced = Fedot(ml_task=task_type,
                           composer_params=composer_params)

    model_advanced.fit(features=train)
    model_advanced.predict(features=test)

    metric_advanced = model_advanced.quality_metric()
    threshold = np.std(pd.read_csv(test)['target'].values)

    assert metric_advanced < threshold