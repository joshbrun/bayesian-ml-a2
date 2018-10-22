# coding=utf-8

"""
SE755 - A2
Data set: 3 - Landsat Satellite Data
ML type: Classification

Algorithms: logistic_regression,
Runs the training and analysis of the above three classification algorithm

Authors:
Joshua Brundan
Kevin Hira
"""

def run(options):
    pass


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
from src.classification_landsat.preprocess import preprocess
from src.classification_landsat.train_logistic_regression import train as train_logistic_regression
import os
from sklearn.externals import joblib
import sys

DIR = os.path.join(os.getcwd(), "src", "classification_landsat")
PATH = os.path.join(DIR, "landsat.csv")
ANALYSIS_PATH = os.path.join(DIR, "analysis")

ANALYSIS_FILE = os.path.join(ANALYSIS_PATH, "log_reg.csv")


def run(options):
    # Get the data
    data = load(PATH, False)
    analysing = options['analysis']
    training = options['train']
    evaluating_data_path = options['newdata']

    if analysing:
        # Data pre-processing
        if not os.path.isdir(ANALYSIS_PATH):
            os.mkdir(ANALYSIS_PATH)
        delete_file(ANALYSIS_FILE)

        preprocessed_data = preprocess(data, False, True)

        # Train the data
        # model, training_analysis
        append_to_file(ANALYSIS_FILE, train_logistic_regression(preprocessed_data, True, False))
        print()

    if training:
        # train the model
        preprocessed_data = preprocess(data, True)
        model = train_logistic_regression(preprocessed_data, True, False)

        # Save model
        if not os.path.isdir("models"):
            os.mkdir('models')
        joblib.dump(model, 'models/log_reg_model.joblib')

    else:
        if not os.path.isdir("models"):
            print("Model has not been trained")
            print("\trun:  python run.py -t -d N")
            print("\twhere N is the dataset you are evaluating on.")
            exit(2)
        model = joblib.load('models/log_reg_model.joblib')
    if evaluating_data_path is not None:

        path = os.path.join(os.getcwd(), evaluating_data_path)
        evaluating_data = load(path, False)
        preprocessed_eval_data = preprocess(data, True, evaluating_data)
        return [str(x) for x in model.predict(preprocessed_eval_data)]
