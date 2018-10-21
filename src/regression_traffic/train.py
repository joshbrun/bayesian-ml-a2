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


def train(data, analysis):
    """
    Trains the bayesian linear regression model
    :param data:
    :return:
    """

    features, target = split_input_and_target(data)
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.10)

    # Calculate base line

    base_line_predictions = pandas.DataFrame({'prediction':[42]*6750})
    base_mae, base_rmse = regression_prediction_evaluation(base_line_predictions.values, y_train.values)

    reg = None
    if analysis:
        model = BayesianRidge()

        tuned_parameters = [{'alpha_1': [1e-5, 1e-6, 1e-7],
                             'alpha_2': [1e-5, 1e-6, 1e-7],
                             'lambda_1': [1e-5, 1e-6, 1e-7],
                             'lambda_2': [1e-5, 1e-6, 1e-7]}]
        searcher = GridSearchCV(model, tuned_parameters, cv=15)
        searcher.fit(x_train, y_train)

        reg = searcher.best_estimator_

        means = searcher.cv_results_['mean_test_score']
        stds = searcher.cv_results_['std_test_score']

        # Testing data
        testing_true, testing_pred = y_test, reg.predict(x_test)
        testing_mae, testing_rmse = regression_prediction_evaluation(testing_true, testing_pred)

        # Training data
        training_true, training_pred = y_train, reg.predict(x_train)
        training_mae, training_rmse = regression_prediction_evaluation(training_true, training_pred)

        # Calculations
        training_vs_baseline_mae = 100 * abs(training_mae - base_mae) / base_mae
        testing_vs_baseline_mae = 100 * abs(testing_mae - base_mae) / base_mae
        training_vs_testing_mae = 100 * abs(training_mae - testing_mae) / testing_mae

        training_vs_baseline_rmse = 100 * abs(training_rmse - base_rmse) / base_rmse
        testing_vs_baseline_rmse = 100 * abs(testing_rmse - base_rmse) / base_rmse
        training_vs_testing_rmse = 100 * abs(training_rmse - testing_rmse) / testing_rmse

        return training_vs_baseline_mae, testing_vs_baseline_mae, training_vs_testing_mae, training_vs_baseline_rmse, testing_vs_baseline_rmse, training_vs_testing_rmse



    else:
        reg = BayesianRidge(alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6)
        reg.fit(x_train, y_train)

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


def regression_prediction_evaluation(predictions, true):

    mae = numpy.mean(abs(predictions - true))
    rmse = numpy.sqrt(numpy.mean((predictions - true) ** 2))

    return mae, rmse

