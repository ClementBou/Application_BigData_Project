import pandas as pd


def handling_dataframe(dataframe, stable=False):
    """
    return a sample of a cleaned given dataset
    :param stable: boolean to know if the target need to be stable
    :param dataframe: a dataset to clear
    :return: sample of a cleared dataset
    """
    # To heavy dataset so we only use a sub dataset
    if stable:
        dataframe = dataframe.groupby("TARGET").sample(n=15000, random_state=1).sample(frac=1, random_state=1)
    else:
        dataframe = dataframe.sample(n=30000, random_state=1)

    # Fill missed data
    dataframe.fillna(dataframe.mean(), inplace=True)

    # Drop duplicated data
    dataframe.drop_duplicates(inplace=True)

    return dataframe


def data_preparation(dataset_train, dataset_eval):
    """Get data set from a csv path and clear them.

    :param dataset_train: path of training dataset.
    :type dataset_train: str.
    :param dataset_eval: path of evaluating dataset.
    :type dataset_eval: str.
    :return: train_data : training data in on Dataframe.
    :rtype: Pandas Dataframe.
    :return: eval_data : evaluating data in on Dataframe.
    :rtype: Pandas Dataframe.
    """

    train_data = handling_dataframe(pd.read_csv(dataset_train), stable=True)
    eval_data = handling_dataframe(pd.read_csv(dataset_eval))
    return train_data, eval_data
