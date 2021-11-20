import sys

import mlflow
import xgboost as xgb
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from src.data_interpretation import get_interpretation


def train_Xgboost(X_train, y_train):
    """Train a Xgboost model with training dataset.

    :param X_train: chosen data set to train the model.
    :type X_train: Pandas Dataset.
    :param y_train: chosen target to train the model.
    :type y_train: Pandas Dataset.
    :return model : Xgboost
    """
    mlflow.xgboost.autolog()

    num_class = int(sys.argv[1]) if len(sys.argv) < 1 else 1
    learning_rate = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    model = xgb.XGBClassifier(num_class=num_class, learning_rate=learning_rate)
    model.fit(X_train, y_train)

    mlflow.sklearn.log_model(model, "xgboost")
    return model


def train_Random_Forest(X_train, y_train):
    """Train a Random Forest model with training dataset.

    :param X_train: chosen data set to train the model.
    :type X_train: Pandas Dataset.
    :param y_train: chosen target to train the model.
    :type y_train: Pandas Dataset.
    :return model : RandomForestClassifier
    """

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    mlflow.sklearn.log_model(model, "random_forest")
    return model


def train_Gradient_Boosting(X_train, y_train):
    """Train a Gradient Boosting model with training dataset.

    :param X_train: chosen data set to train the model.
    :type X_train: Pandas Dataset.
    :param y_train: chosen target to train the model.
    :type y_train: Pandas Dataset.
    :return model : GradientBoostingClassifier
    """

    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)

    mlflow.sklearn.log_model(model, "gradient_boosting")
    return model


def models_training(models_name, train_data, interpretations):
    """Training of model chosen with training dataset.

    :param models_name:  name of the chosen model.
    :type models_name: str.
    :param train_data: data set to train the model.
    :type train_data: Pandas Dataframe.
    :return: model:  trained model.
    :rtype: model : model.
    :raises: AttributeError, KeyError.
    """

    X_train, X_test, y_train, y_test = train_test_split(train_data.drop(['TARGET'], axis=1), train_data['TARGET'], random_state=1)

    for model_name in models_name:
        print("Result for : " + model_name)

        if model_name == "Xgboost":
            model = train_Xgboost(X_train, y_train)
        elif model_name == "Random Forest":
            model = train_Random_Forest(X_train, y_train)
        elif model_name == "Gradient Boosting":
            model = train_Gradient_Boosting(X_train, y_train)
        else:
            raise ValueError("This model isn't available. Try \"Xgboost\" or \"Random Forest\" or \"Gradient Boosting\"")

        print(model.score(X_test, y_test))

        # evaluate model
        y_proba = model.predict(X_test)
        loss = log_loss(y_test, y_proba)
        acc = accuracy_score(y_test, y_proba)

        # log metrics
        mlflow.log_metrics({"log_loss": loss, "accuracy": acc})

        # get interpretation
        get_interpretation(model, X_test, interpretations)