# coding=utf-8

"""
SE755 - A2
Trains the regression model

Authors:
Joshua Brundan
Kevin Hira
"""

import pandas
from sklearn.model_selection import train_test_split
import numpy
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model.logistic import LogisticRegressionCV
from sklearn.metrics import classification_report


def train(data, train, analysis):
    """
    Trains the bayesian linear regression model
    :param data:
    :return:
    """

    if analysis or train:
        data = data.drop(columns=['index'])

        features, target = split_input_and_target(data)
        x_train, x_test, y_train, y_test = train_test_split(features, target, stratify=target, test_size=0.10)
        # No need for GridSearch, as no hyperparameters

        clf = LogisticRegressionCV(cv=5, max_iter=2500, multi_class='auto')

        print("This will take a short while to fit... Please ignore the DataConversionWarning below:")
        clf.fit(x_train, y_train)

        train_pred = clf.predict(x_train)

        test_pred = clf.predict(x_test)

        if not analysis:
            print("Training Classification Report:")
            print(classification_report(y_train, train_pred))

            print(classification_report(y_test, test_pred))

            print("\nTesting Classification Report:")
            print(clf.score(x_train, y_train))
            print(clf.score(x_test, y_test))
        return clf

def split_input_and_target(data):
    """
    Split the feature and target columns
    :param data: The combined dataframe
    :return: features columns and target columns
    """

    features = data.iloc[:,:-1].copy()
    target = data.iloc[:,-1:].copy()

    return features, target

def regression_prediction_evaluation(predictions, true):

    mae = numpy.mean(abs(predictions - true))
    rmse = numpy.sqrt(numpy.mean((predictions - true) ** 2))

    return mae, rmse

