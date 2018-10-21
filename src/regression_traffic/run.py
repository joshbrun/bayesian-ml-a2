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

from src.common.load import load, append_to_file, delete_file
from src.regression_traffic.preprocess import preprocess
from src.regression_traffic.train import train
import os

DIR = os.path.join(os.getcwd(), "src", "regression_traffic")
PATH = os.path.join(DIR, "traffic_flow_data.csv")
ANALYSIS_PATH = os.path.join(DIR, "analysis")

ANALYSIS_FILE = os.path.join(ANALYSIS_PATH, "regression.csv")
ORDERED_FEATURES = os.path.join(ANALYSIS_PATH, "features.csv")


BEST_FEATURE_COUNT = 450
FEATURE_LIST = ["a","b","c"]

def run(analysis):

    # Get the data
    data = load(PATH, True)

    if analysis:
        # Data pre-processing
        results = []

        if not os.path.isdir(ANALYSIS_PATH):
            os.mkdir(ANALYSIS_PATH)

        delete_file(ANALYSIS_FILE)
        delete_file(ORDERED_FEATURES)

        # Feature analysis loops
        # n = number of features/10
        for n in range(10, 460, 10):
            print("n=%d"%(n*10))

            # N is the number of parameters to choose
            preprocessed_data, most_correlated = preprocess(data, n)

            # Train the data
            # model, training_analysis
            append_to_file(ANALYSIS_FILE, train(preprocessed_data, analysis))
            append_to_file(ORDERED_FEATURES, most_correlated)

            # analysis(model, training_analysis)

        for r in results:
            print(r)

        # save the best model

    else:
        # train the model
        pass

