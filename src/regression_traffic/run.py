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
from sklearn.externals import joblib
import sys

DIR = os.path.join(os.getcwd(), "src", "regression_traffic")
PATH = os.path.join(DIR, "traffic_flow_data.csv")
ANALYSIS_PATH = os.path.join(DIR, "analysis")

ANALYSIS_FILE = os.path.join(ANALYSIS_PATH, "regression.csv")
ORDERED_FEATURES = os.path.join(ANALYSIS_PATH, "features.csv")

BEST_FEATURE_COUNT = 10
FEATURE_LIST = ["Segment_22(t)", "Segment_23(t)", "Segment_24(t)", "Segment_21(t)",	"Segment_25(t)", "Segment_22(t-1)", "Segment_21(t-1)", "Segment_24(t-1)", "Segment_23(t-1)", "Segment_16(t)", "Segment23_(t+1)"]


def run(options):

    # Get the data
    data = load(PATH, True)
    analysing = options['analysis']
    training = options['train']
    evaluating_data_path = options['newdata']

    if analysing:
        # Data pre-processing
        results = []

        if not os.path.isdir(ANALYSIS_PATH):
            os.mkdir(ANALYSIS_PATH)

        delete_file(ANALYSIS_FILE)
        delete_file(ORDERED_FEATURES)

        # Feature analysis loops
        # n = number of features/10
        for n in range(10, 460, 10):
            sys.stdout.write("\r%d%%" % (n*100/450))
            sys.stdout.flush()

            # N is the number of parameters to choose
            preprocessed_data, most_correlated = preprocess(data, n, None)

            # Train the data
            # model, training_analysis
            append_to_file(ANALYSIS_FILE, train(preprocessed_data, analysing))
            append_to_file(ORDERED_FEATURES, most_correlated)
        print()

        for r in results:
            print(r)

    if training:
        # train the model
        preprocessed_data, most_correlated = preprocess(data, BEST_FEATURE_COUNT, FEATURE_LIST)
        model = train(preprocessed_data, analysing)

        # Save model
        joblib.dump(model, 'blr_model.joblib')

    else:
        model = joblib.load('blr_model.joblib')

    if evaluating_data_path is not None:

        path = os.path.join(os.getcwd(), evaluating_data_path)
        evaluating_data = load(path, True)
        preprocessed_eval_data = preprocess(data, BEST_FEATURE_COUNT, FEATURE_LIST, evaluating_data)
        return [str(x) for x in model.predict(preprocessed_eval_data)]
