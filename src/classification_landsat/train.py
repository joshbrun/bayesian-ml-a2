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


def train(data):
    """
    Trains the bayesian linear regression model
    :param data:
    :return:
    """

    features, target = split_input_and_target(data)
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.10)



    # Calculate base line

    # base_line_predictions = pandas.DataFrame({'prediction':[42]*6750})
    # mae, rmse = regression_prediction_evaluation(base_line_predictions.values, y_train.values)

    # print('Median Baseline  MAE: {:.4f}'.format(mae))
    # print('Median Baseline RMSE: {:.4f}'.format(rmse))

    # model = BayesianRidge()

    # tuned_parameters = [{'alpha_1': [1e-5, 1e-6, 1e-7],
    #                      'alpha_2': [1e-5, 1e-6, 1e-7],
    #                      'lambda_1': [1e-5, 1e-6, 1e-7],
    #                      'lambda_2': [1e-5, 1e-6, 1e-7]}]

    # tuned_parameters = [{'alpha_1': [1e-5, 1e-6]}]
    #
    # reg = GridSearchCV(model, tuned_parameters, cv=15)
    #
    # reg.fit(x_train, y_train)

    # print("Best parameters set found on development set:")
    # print()
    # print(reg.best_params_)
    # print()
    # print("Grid scores on development set:")
    # print()
    # means = reg.cv_results_['mean_test_score']
    # stds = reg.cv_results_['std_test_score']
    # for mean, std, params in zip(means, stds, reg.cv_results_['params']):
    #     print("%0.3f (+/-%0.03f) for %r"
    #           % (mean, std * 2, params))
    # print()
    #
    # print("Detailed classification report:")
    # print()
    # print("The model is trained on the full development set.")
    # print("The scores are computed on the full evaluation set.")
    # print()



    # Testing data
    # testing_true, testing_pred = y_test, reg.predict(x_test)
    # mae2, rmse2 = regression_prediction_evaluation(testing_true, testing_pred)
    # print()
    # print('Median Testing  MAE: {:.4f}'.format(mae2))
    # print('Median Testing RMSE: {:.4f}'.format(rmse2))

    # Training data
    # training_true, training_pred = y_train, reg.predict(x_train)
    # mae3, rmse3 = regression_prediction_evaluation(training_true, training_pred)

    # print('Median Training  MAE: {:.4f}'.format(mae3))
    # print('Median Training RMSE: {:.4f}'.format(rmse3))

    # training_vs_baseline_mae = 100*abs(mae3-mae)/mae
    # testing_vs_baseline_mae = 100*abs(mae2-mae)/mae
    # training_vs_testing_mae = 100*abs(mae3-mae2)/mae2

    # print("training_vs_baseline_mae {:0.2f}%".format(training_vs_baseline_mae))
    # print("testing_vs_baseline_mae {:0.2f}%".format(testing_vs_baseline_mae))
    # print("training_vs_testing_mae {:0.2f}%".format(training_vs_testing_mae))
    # return mae, rmse, mae2, rmse2, mae3, rmse3, training_vs_baseline_mae, testing_vs_baseline_mae, training_vs_testing_mae


def split_input_and_target(data):
    """
    Split the feature and target columns
    :param data: The combined dataframe
    :return: features columns and target columns
    """

    features = data.iloc[:,:-1].copy()
    target = data.iloc[:,-1].copy()

    return features, target


def regression_prediction_evaluation(predictions, true):

    mae = numpy.mean(abs(predictions - true))
    rmse = numpy.sqrt(numpy.mean((predictions - true) ** 2))

    return mae, rmse

