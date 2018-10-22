# coding=utf-8

"""
SE755 - A2
Determines the success of a regression based model

Authors:
Joshua Brundan
Kevin Hira
"""

from sklearn.metrics import adjusted_rand_score, completeness_score, homogeneity_score, fowlkes_mallows_score


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

    metrics = [
                {
                    'name' : 'Adjusted Rand Score',
                    'func' : adjusted_rand_score,
                    'test_score' : 0,
                    'train_score' : 0,
                    'overfit_score' : 0
                },
                {
                    'name' : 'Completeness Score',
                    'func' : completeness_score,
                    'test_score' : 0,
                    'train_score' : 0,
                    'overfit_score' : 0
                },
                {
                    'name' : 'Homogeneity Score',
                    'func' : homogeneity_score,
                    'test_score' : 0,
                    'train_score' : 0,
                    'overfit_score' : 0
                },
                {
                    'name' : 'Fowlkes-Mallows Score',
                    'func' : fowlkes_mallows_score,
                    'test_score' : 0,
                    'train_score' : 0,
                    'overfit_score' : 0
                }
            ]

    data = []

    for metric in metrics:
        metric['test_score'] = metric['func'](testing_actual, testing_prediction)
        metric['train_score'] = metric['func'](training_actual, training_prediction)
        metric['overfit_score'] = percentage_improvement(metric['test_score'], metric['train_score'])
        if verbose:
            print(metric['name'])
            print('\tTest Set:\t\t%f' % (metric['test_score']))
            print('\tTraining Set:\t\t%f' % (metric['train_score']))
            print('\tOverfitting:\t\t%f%%' % (metric['overfit_score']))

    data = [(metric['test_score'], metric['train_score'], metric['overfit_score']) for metric in metrics]
    return [i for j in data for i in j]
