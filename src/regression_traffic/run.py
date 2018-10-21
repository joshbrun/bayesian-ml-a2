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
from src.regression_traffic.preprocess import preprocess
from src.regression_traffic.train import train
import os

DATA_PATH = os.path.join(os.getcwd(), "src", "regression_traffic", "traffic_flow_data.csv")

def run():

    # Get the data
    data = load(DATA_PATH, True)

    # Data pre-processing
    results = []

    # Feature analysis loops
    # n = number of features/10
    for n in range(10, 460, 10):
        print("n=%d"%(n*10))

        # N is the number of parameters to choose
        preprocessed_data, most_correlated = preprocess(data, n)



        # Train the data
        # model, training_analysis
        results.append(train(preprocessed_data))

        # analysis(model, training_analysis)

    for r in results:
        print(r)

    # save the best model

