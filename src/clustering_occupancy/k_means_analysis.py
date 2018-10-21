# coding=utf-8

"""
SE755 - A2
Determines the success of a regression based model

Authors:
Joshua Brundan
Kevin Hira
"""

from sklearn.metrics import adjusted_rand_score


def percentage_improvement(a, b):
    if b == 0:
        b = 0.00001
    return 100 * abs(a - b) / b


def analysis_model(testing_actual, testing_prediction, training_actual, training_prediction,
                   verbose=False):
    """
    Creates a report indicating the success of the model.
    :param baseline_prediction: an array of a random number, to form a baseline
    :return:
    """

    testing_rand = adjusted_rand_score(testing_actual, testing_prediction)
    training_rand = adjusted_rand_score(training_actual, training_prediction)

    overfitting_percentage = percentage_improvement(testing_rand, training_rand)

    output = [testing_rand, training_rand, overfitting_percentage]

    if verbose:
        print(output)
    
    return output
