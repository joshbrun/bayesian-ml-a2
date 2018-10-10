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


def preprocess(raw_data):
    """
    preprocess the traffic data
    :param data: the raw traffic data
    :return: the preprocessed traffic data
    """

    data = feature_extraction(raw_data)
    print(data)
    print(data.head())

    # Keep all the traffic features

    # Split data into features and target
    features, target = split_input_and_target(data)


    # Normalise the columns
    normalised_features = normalise(features)

    preprocessed_data = pandas.concat([normalised_features, target], axis=1)

    return preprocessed_data


def normalise(data):
    """
    Normalise all numerical columns between the range [0,1]
    :param data:
    :return:
    """
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

def feature_extraction(data):

    # return data

    most_correlated = data.corr().abs()['Segment23_(t+1)'].sort_values(ascending=False)

    # Get the top 5 coorelated rows

    # Compare different number N down there
    n=100
    best_columns = most_correlated[1:n+1].index.values
    best_columns = numpy.append(best_columns, ['Segment23_(t+1)'])
    print(best_columns)
    return data.loc[:, best_columns].copy()




