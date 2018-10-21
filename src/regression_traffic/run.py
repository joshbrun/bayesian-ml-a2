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

DATA_PATH="traffic_flow_data.csv"

def run():

    # Get the data
    data = load(DATA_PATH, True)

    # Data pre-processing
    results = []
    for n in range(45):
        print("n=%f"%(n*10))
        if n == 0:
            continue

        # N is the number of parameters to choose
        preprocessed_data = preprocess(data, n*10)

        # Train the data
        # model, training_analysis
        results.append(train(preprocessed_data))

        # analysis(model, training_analysis)

    for r in results:
        print(r)

run()