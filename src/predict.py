import os
import mlflow


def predict(models_path, eval_data):
    """Ask a model to predict target feature on an evaluating dataset.

    :param models_path: models path.
    :type models_path: str.
    :param eval_data: evaluating dataset.
    :type eval_data: Pandas Dataframe.
    :return: prediction : model predictions.
    :rtype: prediction: Pandas DataFrame.
    """

    for model_path in os.listdir(models_path):
        print("Predictions for : " + model_path)
        model = mlflow.sklearn.load_model(models_path + '/' + model_path)
        predictions = model.predict(eval_data)
        print(predictions)
