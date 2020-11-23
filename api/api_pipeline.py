from api.run_api import fedot_runner
import pandas as pd
import os
from fedot.core.utils import project_root
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    task_type = 'reg'
    file_path_train = 'cases/data/oil_chemistry/train.csv'
    file_path_test = 'cases/data/oil_chemistry/test_1.csv'
    composer_params = {'max_depth': 2,
                       'max_arity': 2,
                       'pop_size': 10,
                       'num_of_generations': 10,
                       'learning_time': 3}

    # create paths to our datasets.
    full_path_train = os.path.join(str(project_root()), file_path_train)
    full_path_test = os.path.join(str(project_root()), file_path_test)

    # create a train and test datasets. Our framework allows to use both type of data - csv files or Numpy arrays.
    train_file = pd.read_csv(full_path_train)
    X, y = train_file.loc[:, ~train_file.columns.isin(['target'])].values, train_file['target'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=24)

    # run a fedot operator with basic composer settings.
    model_baseline = fedot_runner(ml_task=task_type)
    fitted_model_baseline = model_baseline.fit(features=X_train,
                                               target=y_train)
    prediction_baseline = model_baseline.predict(features=X_test,
                                                 target=y_test)
    metric_baseline = model_baseline.quality_metric()
    print(metric_baseline)
    print(prediction_baseline)

    # run a fedot operator with customized composer settings.
    model_advanced = fedot_runner(ml_task=task_type,
                                  composer_params=composer_params)
    fitted_model_advanced = model_advanced.fit(csv_path=full_path_train)
    prediction_advanced = model_advanced.predict(csv_path=full_path_test)
    metric_advanced = model_advanced.quality_metric()

    print(metric_advanced)
    print(prediction_advanced)
