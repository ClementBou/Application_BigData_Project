import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pickle


def train_Xgboost(train_data):
    """Train a Xgboost model with training dataset.

    :param train_data: chosen data set to train the model.
    :type train_data: Pandas Dataset.
    """

    X_train, X_test, y_train, y_test = train_test_split(train_data.drop(['TARGET'], axis=1), train_data['TARGET'], random_state=1)

    model = xgb.XGBClassifier()

    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))

    model.save_model("xgboost.json")


def train_Random_Forest(train_data):
    """Train a Random Forest model with training dataset.

    :param train_data: chosen data set to train the model.
    :type train_data: Pandas Dataset.
    """

    X_train, X_test, y_train, y_test = train_test_split(train_data.drop(['TARGET'], axis=1), train_data['TARGET'], random_state=1)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))

    with open('random_forest.pkl', 'wb') as f:
        pickle.dump(model, f)


def train_Gradient_Boosting(train_data):
    """Train a Gradient Boosting model with training dataset.

    :param train_data: chosen data set to train the model.
    :type train_data: Pandas Dataset.
    """

    X_train, X_test, y_train, y_test = train_test_split(train_data.drop(['TARGET'], axis=1), train_data['TARGET'], random_state=1)

    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))

    with open('gradient_boosting.pkl', 'wb') as f:
        pickle.dump(model, f)


def models_training(models_name, train_data):
    """Training of model chosen with training dataset.

    :param models_name:  name of the chosen model.
    :type models_name: str.
    :param train_data: data set to train the model.
    :type train_data: Pandas Dataframe.
    :return: model:  trained model.
    :rtype: model : model.
    :raises: AttributeError, KeyError.
    """
    for model_name in models_name:
        print("Result for : " + model_name)
        if model_name == "Xgboost":
            train_Xgboost(train_data)
        elif model_name == "Random Forest":
            train_Random_Forest(train_data)
        elif model_name == "Gradient Boosting":
            train_Gradient_Boosting(train_data)
        else:
            raise ValueError("This model isn't available. Try \"Xgboost\" or \"Random Forest\" or \"Gradient Boosting\"")