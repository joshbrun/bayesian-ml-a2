# coding=utf-8

"""
SE755 - A2
Data set: 3 - Landsat Satellite Data
ML type: Classification

Algorithms: logistic_regression, deep_learning
Runs the training and analysis of the above three classification algorithms

Authors:
Joshua Brundan
Kevin Hira
"""

from src.common.load import load
from src.classification_landsat.preprocess import preprocess
from src.classification_landsat.train import train
import matplotlib.pyplot as plt

DATA_PATH="lantsat.csv"

def run():

    # Get the data
    data = load(DATA_PATH, False)

    plt.hist(data[36], bins=5)
    plt.show()


    # Data pre-processing
    preprocessed_data = preprocess(data)
    print(preprocessed_data)
    # Train the data
    # model, training_analysis
    # train(preprocessed_data)

    # analysis(model, training_analysis)

run()