def predict(model, eval_data):
    """Ask a model to predict target feature on an evaluating dataset.

    :param model: trained model to ask.
    :type model: model.
    :param eval_data: evaluating dataset.
    :type eval_data: Pandas Dataframe.
    :return: prediction : model predictions.
    :rtype: prediction: Pandas DataFrame.
    """
    return