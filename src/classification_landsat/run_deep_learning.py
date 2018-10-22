# coding=utf-8

"""
SE755 - A2
Data set: 3 - Landsat Satellite Data
ML type: Classification

Algorithms: deep_learning
Runs the training and analysis of the above three classification algorithm

Authors:
Joshua Brundan
Kevin Hira
"""
from src.common.load import load, append_to_file, delete_file
from src.classification_landsat.preprocess import preprocess
from src.classification_landsat.train_deep_learning import train as train_deep_learning
import os
from sklearn.externals import joblib
import sys
import tensorflow as tf

DIR = os.path.join(os.getcwd(), "src", "classification_landsat")
PATH = os.path.join(DIR, "landsat.csv")
ANALYSIS_PATH = os.path.join(DIR, "analysis")

ANALYSIS_FILE = os.path.join(ANALYSIS_PATH, "deep_reg.csv")

BEST_HIDDEN_LAYER = [500, 250, 50]

def run(options):

    print("Tensorflow verision: "+tf.__version__)
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

        preprocessed_data = preprocess(data, False)

        hidden_layer_1 = [20,50,100]
        hidden_layer_2 = [50,100,250,500]
        hidden_layer_3 = [50,250,500,750]
        # Train the data
        # model, training_analysis

        # Structural Analysis
        print("Running - This will take a long while...")
        print("Please ignore the tensorflow warnings below:")
        for i in hidden_layer_3:
            for j in hidden_layer_2:
                for k in hidden_layer_1:
                    append_to_file(ANALYSIS_FILE, train_deep_learning(preprocessed_data, True, [i, j, k]))

    if evaluating_data_path is not None:
        preprocessed_data = preprocess(data, True)
        path = os.path.join(os.getcwd(), evaluating_data_path)
        evaluating_data = load(path, False)
        preprocessed_eval_data = preprocess(data, True, evaluating_data)
        return train_deep_learning(preprocessed_data, False, BEST_HIDDEN_LAYER, True, preprocessed_eval_data, True)
