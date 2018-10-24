# coding=utf-8

"""
SE755 - A2
Determines the success of a regression based model

Authors:
Joshua Brundan
Kevin Hira
"""

from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error


def percentage_improvement(a, b):
    if b == 0:
        b = 0.00001
    return 100 * abs(a - b) / b


def analysis_model(testing_base_line_predictions, training_base_line_predictions, testing_actual, testing_prediction, training_actual, training_prediction,
                   verbose=False):
    """
    Creates a report indicating the success of the model.
    :param baseline_prediction: an array of a random number, to form a baseline
    :param testing_actual: The correct results if 0 error
    :param testing_prediction: The predicted results from the model
    :param training_actual: The correct results if 0 error
    :param training_prediction: The predicted results from the model
    :return:
    """

    testing_variance_score = explained_variance_score(testing_actual, testing_prediction)
    training_variance_score = explained_variance_score(training_actual, training_prediction)

    if verbose:
        print("-" * 90)
        print("Data set 1 error/accuracies:")
        print()
        print("Variance scores:")
        print("testing_variance_score: %f" % testing_variance_score)
        print("training_variance_score: %f" % training_variance_score)
        print()

    testing_vs_basecase_mae_score = mean_absolute_error(testing_actual, testing_base_line_predictions)
    testing_mae_score = mean_absolute_error(testing_actual, testing_prediction)
    testing_improvement_mae = percentage_improvement(testing_mae_score, testing_vs_basecase_mae_score)

    training_vs_basecase_mae_score = mean_absolute_error(training_actual, training_base_line_predictions)
    training_mae_score = mean_absolute_error(training_actual, training_prediction)
    training_improvement_mae = percentage_improvement(training_mae_score, training_vs_basecase_mae_score)

    overfitting_from_mae = percentage_improvement(testing_vs_basecase_mae_score, training_vs_basecase_mae_score)
    overfitting_from_basecase_mae = percentage_improvement(testing_mae_score, training_mae_score)

    if verbose:
        print()
        print("MAE scores:")
        print("testing_vs_basecase_mae_score: %f" % testing_vs_basecase_mae_score)
        print("testing_mae_score: %f" % testing_mae_score)
        print("training_vs_basecase_mae_score: %f" % training_vs_basecase_mae_score)
        print("training_mae_score: %f" % training_mae_score)
        print()
        print("Model improvement from basecase:")
        print("testing_improvement_mae %f%%" % testing_improvement_mae)
        print("training_improvement_mae %f%%" % training_improvement_mae)
        print()
        print("Overfitting:")
        print("Basecase: %f%%" % overfitting_from_basecase_mae)
        print("Variance: %f%%" % overfitting_from_mae)
        print()

    testing_vs_basecase_mse_score = mean_squared_error(testing_actual, testing_base_line_predictions)
    testing_mse_score = mean_squared_error(testing_actual, testing_prediction)
    testing_improvement_mse = percentage_improvement(testing_mse_score, testing_vs_basecase_mse_score)

    training_vs_basecase_mse_score = mean_squared_error(training_actual, training_base_line_predictions)
    training_mse_score = mean_squared_error(training_actual, training_prediction)
    training_improvement_mse = percentage_improvement(training_mse_score, training_vs_basecase_mse_score)

    overfitting_from_mse = percentage_improvement(testing_vs_basecase_mse_score, training_vs_basecase_mse_score)
    overfitting_from_basecase_mse = percentage_improvement(testing_mse_score, training_mse_score)

    if verbose:
        print()
        print("MSE scores:")
        print("testing_vs_basecase_mse_score: %f" % testing_vs_basecase_mse_score)
        print("testing_mse_score: %f" % testing_mse_score)
        print("training_vs_basecase_mse_score: %f" % training_vs_basecase_mse_score)
        print("training_mse_score: %f" % training_mse_score)
        print()
        print("Model improvement from basecase:")
        print("testing_improvement_mse %f%%" % testing_improvement_mse)
        print("training_improvement_mse %f%%" % training_improvement_mse)
        print()
        print("Indicator of Overfitting:")
        print("Basecase: %f%%" % overfitting_from_basecase_mse)
        print("Variance: %f%%" % overfitting_from_mse)
        print()

    output = [testing_variance_score, training_variance_score,

              testing_vs_basecase_mae_score, testing_mae_score, testing_improvement_mae,
              training_vs_basecase_mae_score, training_mae_score, training_improvement_mae,
              overfitting_from_mae, overfitting_from_basecase_mae,

              testing_vs_basecase_mse_score, testing_mse_score, testing_improvement_mse,
              training_vs_basecase_mse_score, training_mse_score, training_improvement_mse,
              overfitting_from_mse, overfitting_from_basecase_mse]

    return output
