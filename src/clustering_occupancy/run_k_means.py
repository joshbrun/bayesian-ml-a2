# coding=utf-8

"""
SE755 - A2
Run the k_means algorithm

Authors:
Joshua Brundan
Kevin Hira
"""

from src.common.load import load, append_to_file, delete_file
from src.clustering_occupancy.preprocess import preprocess
from src.clustering_occupancy.train_k_means import train
import os
from sklearn.externals import joblib
import sys

DIR = os.path.join(os.getcwd(), "src", "clustering_occupancy")
PATH = os.path.join(DIR, "occupancy_sensor_data.csv")
ANALYSIS_PATH = os.path.join(DIR, "analysis")

ANALYSIS_FILE = os.path.join(ANALYSIS_PATH, "k_means_clustering.csv")
ORDERED_FEATURES = os.path.join(ANALYSIS_PATH, "k_means_features.csv")


BEST_FEATURE_COUNT = 1
FEATURE_LIST = ['Light','Occupancy']

def run(options):

    # Get the data
    data = load(PATH, True)
    analysing = options['analysis']
    training = options['train']
    evaluating_data_path = options['newdata']


    if analysing:

        if not os.path.isdir(ANALYSIS_PATH):
            os.mkdir(ANALYSIS_PATH)

        delete_file(ANALYSIS_FILE)
        delete_file(ORDERED_FEATURES)

        # Feature analysis loops
        # n = number of features/10
        for n in range(1, 6):
            sys.stdout.write("\r%d%%" % (n*100/6))
            sys.stdout.flush()

            # N is the number of parameters to choose
            preprocessed_data, most_correlated = preprocess(data, n, None)

            # Train the data
            # model, training_analysis
            append_to_file(ANALYSIS_FILE, train(preprocessed_data, analysing))
            append_to_file(ORDERED_FEATURES, most_correlated)

            sys.stdout.write("\r%d%%" % ((n+1)*100/6))
            sys.stdout.flush()
        print()

    if training:
        # train the model
        preprocessed_data, most_correlated = preprocess(data, BEST_FEATURE_COUNT, FEATURE_LIST)
        model = train(preprocessed_data, analysing)

        # Save model
        if not os.path.isdir("models"):
            os.mkdir('models')
        joblib.dump(model, 'models/k_means_model.joblib')

    else:
        if not os.path.isdir("models"):
            print("Model has not been trained")
            print("\trun:  python run.py -t -d N")
            print("\twhere N is the dataset you are evaluating on.")
            exit(2)
        model = joblib.load('models/k_means_model.joblib')

    if evaluating_data_path is not None:

        path = os.path.join(os.getcwd(), evaluating_data_path)
        evaluating_data = load(path, True)
        preprocessed_eval_data = preprocess(data, BEST_FEATURE_COUNT, FEATURE_LIST, evaluating_data)
        return [str(x) for x in model.predict(preprocessed_eval_data)]
