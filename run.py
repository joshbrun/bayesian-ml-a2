# coding=utf-8

"""
SE755 - A2
The run script for SE755 A2

Authors:
Joshua Brundan
Kevin Hira
"""

# Classification Algorithms
from src.classification_landsat.run_logistic_regression import run as logistic_regression_run
from src.classification_landsat.run_deep_learning import run as deep_learning_run

# Regression Algorithm
from src.regression_traffic.run import run as regression_run

# Clustering Algorithm
from src.clustering_occupancy.run import run as clustering_run
from src.clustering_occupancy.run_k_means import run as k_means_run

# Command Line Interface
from src.common.commandinterface import CommandInterface

def horizontal_line():
    """
    prints a simple line of -'s
    """
    print("-"*90)

def print_info():
    """
    Author information appended to the beginning of the terminal output
    """
    horizontal_line()
    print("SE755 A2: Special Topics - Bayesian Machine Learning")
    print("Authors: Joshua Brundan and Kevin Hira")
    horizontal_line()

print_info()
# Accept user arguments
user_interface = CommandInterface()
options = user_interface.get_options()

if options is not None:
    horizontal_line()

    run = {1: regression_run,
           2: k_means_run,
           3: clustering_run,
           4: logistic_regression_run,
           5: deep_learning_run}

    names = {1: 'Bayesian Linear Regression',
           2: 'kMeans Clustering',
           3: 'GMM Clustering',
           4: 'Logistic Regression',
           5: 'Deep Learning'}

    # Training
    if options['train']:
        print("Training:")
        for ml_type in options['dataset']:
            print(options['dataset'])
            print("Algorithm: %d: %s" % (ml_type, names[ml_type]))

            run[ml_type](options)
            pass

        horizontal_line()

    if options['newdata']:
        options['train'] = False
        prediction = run[options['dataset'][0]](options)
        print("Model Predictions:")
        for i in range(len(prediction)):
            print("%d: %s" % (i, prediction[i]))
        horizontal_line()


print("\n")



