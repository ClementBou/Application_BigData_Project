import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_revelant_features(dataframe):
    """Determine which column to keep.

    :param dataframe: data frame to analyse.
    :return: columns impacting target feature.
    """

    # Using Pearson Correlation
    cor = dataframe.corr()

    cor_target = abs(cor['TARGET'])

    cor_target = cor_target[cor_target > 0.02]
    return cor_target


def drop_nan_features(dataframe):
    """ Drop columns with to many nan values (here 50%).

    :param dataframe: data frame to clean.
    :return: data frame without useless columns.
    """

    # Drop useless feature
    dataframe.dropna(how='all', inplace=True)
    dataframe.dropna(axis=1, thresh=(len(dataframe) * 50) / 100, inplace=True)

    # Drop columns 0
    dataframe = dataframe.loc[:, (dataframe != 0).any(axis=0)]

    return dataframe


def feature_engineering(train_data, eval_data):
    """ Found relevant features.

    :param train_data: Training dataset.
    :type train_data: Pandas Dataframe.
    :param eval_data: Evaluating dataset.
    :type eval_data: Pandas Dataframe.
    :return train_data: Training dataset.
    :rtype train_data: Pandas Dataframe.
    :return eval_data: Evaluating dataset.
    :rtype eval_data: Pandas Dataframe.
    """

    # Train dataset handling
    train_data = drop_nan_features(train_data)
    train_data = train_data[get_revelant_features(train_data).index.values]

    # Eval dataset handling
    final_columns = list(train_data.columns.values)
    final_columns.remove('TARGET')
    eval_columns = np.array(final_columns)
    eval_data = eval_data[eval_columns]

    return train_data, eval_data


def print_graph_sklearn(dataframe):
    """
    Plot correlation matrix
    :param dataframe:
    """
    plt.figure(len(dataframe.columns))
    cor = dataframe.corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    plt.show()
