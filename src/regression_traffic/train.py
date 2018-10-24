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
from src.regression_traffic.regression_analysis import analysis_model


def train(data, analysis):
    """
    Trains the bayesian linear regression model
    :param data:
    :return:
    """

    features, target = split_input_and_target(data)
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.10)

    # Calculate base line

    training_base_line_predictions = pandas.DataFrame({'prediction': [329]*6750})
    testing_base_line_predictions = pandas.DataFrame({'prediction': [329]*750})

    # base_mae, base_rmse = regression_prediction_evaluation(base_line_predictions.values, y_train.values)

    if analysis:
        model = BayesianRidge()

        tuned_parameters = [{'alpha_1': [1e-5, 1e-6, 1e-7],
                             'alpha_2': [1e-5, 1e-6, 1e-7],
                             'lambda_1': [1e-5, 1e-6, 1e-7],
                             'lambda_2': [1e-5, 1e-6, 1e-7]}]
        searcher = GridSearchCV(model, tuned_parameters, cv=15, n_jobs=-1)
        searcher.fit(x_train, y_train)

        reg = searcher.best_estimator_
        best_parameters = searcher.best_params_

        # Testing data
        testing_true, testing_pred = y_test, reg.predict(x_test)
        # Training data
        training_true, training_pred = y_train, reg.predict(x_train)

        # Calculations
        output = analysis_model(testing_base_line_predictions, training_base_line_predictions, testing_true, testing_pred, training_true, training_pred)
        output.append([best_parameters])
        return output

    else:
        # These values are the best from from the analysis
        reg = BayesianRidge(alpha_1=1e-7, alpha_2=1e-7, lambda_1=1e-5, lambda_2=1e-7)
        reg.fit(x_train, y_train)

        testing_true, testing_pred = y_test, reg.predict(x_test)
        training_true, training_pred = y_train, reg.predict(x_train)

        analysis_model(testing_base_line_predictions, training_base_line_predictions, testing_true, testing_pred, training_true, training_pred, True)


        return reg

def split_input_and_target(data):
    """
    Split the feature and target columns
    :param data: The combined data drame
    :return: features columns and target columns
    """

    features = data.iloc[:,:-1].copy()
    target = data.iloc[:,-1].copy()

    return features, target
