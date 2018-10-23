# coding=utf-8

"""
SE755 - A2
Trains the k_means model

Authors:
Joshua Brundan
Kevin Hira
"""

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from src.clustering_occupancy.clustering_analysis import analysis_model


def train(data, analysis):
    """
    Trains the bayesian linear k_means model
    :param data:
    :return:
    """

    features, target = split_input_and_target(data)
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.10)

    cluster = KMeans(n_clusters=2)
    model = GridSearchCV(cluster, [{'max_iter' : [2500]}], cv=15)
    model.fit(x_train)

    # Testing data
    testing_true, testing_pred = y_test, model.predict(x_test)
    # Training data
    training_true, training_pred = y_train, model.predict(x_train)

    # Calculations
    output = analysis_model(testing_true, testing_pred, training_true, training_pred, not analysis)

    if analysis:
        return output
    else:
        return model

def split_input_and_target(data):
    """
    Split the feature and target columns
    :param data: The combined data drame
    :return: features columns and target columns
    """

    features = data.iloc[:,:-1].copy()
    target = data.iloc[:,-1].copy()

    return features, target
