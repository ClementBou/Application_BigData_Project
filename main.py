import mlflow

from src.data_preparation import data_preparation
from src.feature_engineering import feature_engineering
from src.models_training import models_training
from src.predict import predict

if __name__ == '__main__':
    train_data, eval_data = data_preparation('dataset/application_train.csv', 'dataset/application_test.csv')
    train_data, eval_data = feature_engineering(train_data, eval_data)

    with mlflow.start_run():
        models_training(["Xgboost"], train_data, ["shap_values_all", "shap_values_one", "summary_plot"])
        predict('mlruns/0/' + mlflow.active_run().info.run_id + '/artifacts', eval_data)

