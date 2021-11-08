import pandas as pd


def data_preparation(dataset_train, dataset_eval):
    """Get data set from a csv path.

    :param dataset_train: path of training dataset.
    :type dataset_train: str.
    :param dataset_eval: path of evaluating dataset.
    :type dataset_eval: str.
    :return: train_data : training data in on Dataframe.
    :rtype: Pandas Dataframe.
    :return: eval_data : evaluating data in on Dataframe.
    :rtype: Pandas Dataframe.
    """

    train_data = pd.read_csv(dataset_train)
    eval_data = pd.read_csv(dataset_eval)
    return train_data, eval_data
