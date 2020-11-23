from api.run_api import fedot_runner
import pandas as pd
import os
from core.utils import project_root
from sklearn.model_selection import train_test_split

train_full = r'C:\Users\user\Desktop\FEDOT\cases\data\oil_chemistry\train.csv'
task_type = 'reg'
composer_params = {'max_depth': 2,
                   'max_arity': 2,
                   'pop_size': 10,
                   'num_of_generations': 10,
                   'learning_time': 3}


def get_api_data_paths():
    file_path_train = 'cases/data/oil_chemistry/train.csv'
    file_path_test = 'cases/data/oil_chemistry/test_1.csv'
    full_path_train = os.path.join(str(project_root()), file_path_train)
    full_path_test = os.path.join(str(project_root()), file_path_test)

    return full_path_train, full_path_test


def get_regression_data():
    train_file = pd.read_csv(train_full)
    X, y = train_file.loc[:, ~train_file.columns.isin(['target'])].values, train_file['target'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=24)
    return X_train, X_test, y_train, y_test


def test_api_baseline():
    X_train, X_test, y_train, y_test = get_regression_data()
    model_baseline = fedot_runner(ml_task=task_type)
    model_baseline.fit(features=X_train,
                       target=y_train)
    prediction_baseline = model_baseline.predict(features=X_test,
                                                 target=y_test)
    metric_baseline = model_baseline.quality_metric()
    assert metric_baseline < 0, prediction_baseline.shape == 0


def test_api_advanced():
    train, test = get_api_data_paths()
    model_advanced = fedot_runner(ml_task=task_type,
                                  composer_params=composer_params)
    model_advanced.fit(csv_path=train)
    prediction_advanced = model_advanced.predict(csv_path=test)
    metric_advanced = model_advanced.quality_metric()

    assert metric_advanced < 0, prediction_advanced.shape == 0


test_api_advanced()
