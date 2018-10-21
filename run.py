# coding=utf-8

"""
SE755 - A2
The run script for SE755 A2

Authors:
Joshua Brundan
Kevin Hira
"""

from src.classification_landsat.run import run as classification_run
from src.regression_traffic.run import run as regression_run
from src.clustering_occupancy.run import run as clustering_run

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
           2: clustering_run,
           3: classification_run}



    # Training
    if options['train']:
        print("Training:")
        for ml_type in options['dataset']:
            run[ml_type](options['analysis'])
            pass

        horizontal_line()
    # New Data

    if options['newdata']:
        print("NewData:")
        print("Load %s" % options['newdata'])
        horizontal_line()

        # Load Models

print("\n")



