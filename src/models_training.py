def train_Xgboost(train_data):
    """Train a Xgboost model with training dataset.

    :param train_data: chosen data set to train the model.
    :type train_data: Pandas Dataset.
    :return: model: trained model.
    :rtype: model: ???.
    """
    return


def train_Random_Forest(train_data):
    """Train a Random Forest model with training dataset.

    :param train_data: chosen data set to train the model.
    :type train_data: Pandas Dataset.
    :return: model: trained model.
    :rtype: model: ???.
    """
    return


def train_Gradient_Boosting(train_data):
    """Train a Gradient Boosting model with training dataset.

    :param train_data: chosen data set to train the model.
    :type train_data: Pandas Dataset.
    :return: model: trained model.
    :rtype: model: ???.
    """
    return


def models_training(model_name, train_data):
    """Training of model chosen with training dataset.

    :param model_name:  name of the chosen model.
    :type model_name: str.
    :param train_data: data set to train the model.
    :type train_data: Pandas Dataframe.
    :return: model:  trained model.
    :rtype: model : model.
    :raises: AttributeError, KeyError.
    """

    if model_name == "Xgboost":
        model = train_Xgboost(train_data)
    elif model_name == "Random Forest":
        model = train_Random_Forest(train_data)
    elif model_name == "Gradient Boosting":
        model = train_Gradient_Boosting(train_data)
    else:
        raise ValueError("This model isn't available. Try \"Xgboost\" or \"Random Forest\" or \"Gradient Boosting\"")

    return model
