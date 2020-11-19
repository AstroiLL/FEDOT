from api.run_api import FedotModel

if __name__ == '__main__':
    train_file = r'C:\Users\user\Desktop\FEDOT\cases\data\oil_chemistry\train.csv'
    case_name = 'reg'
    max_time = 2
    fedot = FedotModel(train_file_path=train_file,
                       ml_task=case_name,
                       learning_time=max_time)
    fitted_model = fedot.fit()
    prediction_1 = fedot.predict(test_file=r'C:\Users\user\Desktop\FEDOT\cases\data\oil_chemistry\test_1.csv')
    metric_1 = fedot.quality_metric()
    print(metric_1)
    print(prediction_1)
    prediction_2 = fedot.predict(test_file=r'C:\Users\user\Desktop\FEDOT\cases\data\oil_chemistry\test_2.csv')
    metric_2 = fedot.quality_metric()
    print(metric_2)
    print(prediction_2)
