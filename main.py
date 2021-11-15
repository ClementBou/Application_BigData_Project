from src.data_preparation import data_preparation
from src.feature_engineering import feature_engineering
from src.models_training import models_training
from src.predict import predict

if __name__ == '__main__':

    train_data, eval_data = data_preparation('dataset/application_train.csv', 'dataset/application_test.csv')
    train_data, eval_data = feature_engineering(train_data, eval_data)
    models_training(["Xgboost", "Random Forest", "Gradient Boosting"], train_data)
    predict(["Xgboost", "Random Forest", "Gradient Boosting"], eval_data)

