# coding=utf-8

"""
SE755 - A2
Runs preprocessing on traffic data

Authors:
Joshua Brundan
Kevin Hira
"""

import pandas
from sklearn import preprocessing
import numpy


def preprocess(raw_data, n, feature_list=None, evaluation_data=None):
    """
    preprocess the traffic data
    :param data: the raw traffic data
    :return: the preprocessed traffic data
    """


    if evaluation_data is not None:
        # Add the evaluation_data to the dataset
        raw_data = raw_data.append(evaluation_data, ignore_index=True)

    # n is the number of features to use
    data, most_correlated = feature_extraction(raw_data, n, feature_list)

    # Split data into features and target
    features, target = split_input_and_target(data)

    # Normalise the columns
    normalised_features = normalise(features)

    preprocessed_data = pandas.concat([normalised_features, target], axis=1)

    if evaluation_data is not None:
        data = preprocessed_data.iloc[7500:, :].copy()
        features, target = split_input_and_target(data)
        return features
    else:
        return preprocessed_data, most_correlated



def normalise(data):
    """
    Normalise all numerical columns between the range [0,1]
    :param data:
    :return:
    """
    data = data.astype('float64')

    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(data.values)

    return pandas.DataFrame(x_scaled)

def split_input_and_target(data):
    """
    Split the feature and target columns
    :param data: The combined dataframe
    :return: features columns and target columns
    """

    features = data.iloc[:,:-1].copy()
    target = data.iloc[:,-1].copy()

    return features, target

def feature_extraction(data, n, feature_list):

    # return data

    if feature_list is not None:
        return data.loc[:, feature_list].copy(), feature_list
    else:
        most_correlated = data.corr().abs()['Segment23_(t+1)'].sort_values(ascending=False)
        # Get the top n most coorelated features

        # Compare different number N down there
        best_columns = most_correlated[1:n+1].index.values
        best_columns = numpy.append(best_columns, ['Segment23_(t+1)'])
        return data.loc[:, best_columns].copy(), best_columns




