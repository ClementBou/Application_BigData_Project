import pickle
import xgboost as xgb


def predict(models_name, eval_data):
    """Ask a model to predict target feature on an evaluating dataset.

    :param models_name: trained model to ask.
    :type models_name: model.
    :param eval_data: evaluating dataset.
    :type eval_data: Pandas Dataframe.
    :return: prediction : model predictions.
    :rtype: prediction: Pandas DataFrame.
    """
    for model_name in models_name:
        print("Predictions for : " + model_name)

        if model_name == "Xgboost":
            model = xgb.XGBClassifier()
            model.load_model("xgboost.json")
        elif model_name == "Random Forest":
            with open('random_forest.pkl', 'rb') as f:
                model = pickle.load(f)
        elif model_name == "Gradient Boosting":
            with open('gradient_boosting.pkl', 'rb') as f:
                model = pickle.load(f)
        else:
            raise ValueError("This model isn't available. Try \"Xgboost\" or \"Random Forest\" or \"Gradient Boosting\"")

        predictions = model.predict(eval_data)
        print(predictions)
