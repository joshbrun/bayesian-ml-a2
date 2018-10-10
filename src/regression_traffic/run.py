# coding=utf-8

"""
SE755 - A2
Data set: 1 - H1 Traffic Volume
ML type: Regression

Algorithms: bayesian_linear_regression
Runs the training and analysis of the bayesian linear regression

Authors:
Joshua Brundan
Kevin Hira
"""

from src.common.load import load
DATA_PATH="traffic_flow_data.csv"

def run():

    # Get the data
    data = load(DATA_PATH, True)

    # Data pre-processing
    preprocessed_data = preprocess_data(data)

    # Train the data
    # model, training_analysis = train(preprocessed_data)

    # analysis(model, training_analysis)

    pass

run()