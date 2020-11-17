from api.run_api import FedotModel

if __name__ == '__main__':
    train_file = r'./api_dataset/regression_train.csv'
    test_file = r'./api_dataset/regression_test.csv'
    case_name = 'Regression'
    max_time = 10
    fedot = FedotModel(train_file_path=train_file,
                       test_file_path=test_file,
                       ML_task=case_name,
                       learning_time=max_time)
    prediction = fedot.predict()
    metric = fedot.quality_metric()
    print(prediction)
    print(metric)
