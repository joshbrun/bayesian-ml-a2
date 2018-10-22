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
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, LabelBinarizer
from sklearn.feature_selection import SelectKBest, chi2, f_regression, VarianceThreshold, SelectPercentile, f_classif
import numpy


def preprocess(raw_data, balanced_dataset, user_data=None):
    """
    preprocess the traffic data
    :param data: the raw traffic data
    :return: the preprocessed traffic data
    """

    if user_data is not None:
        # Add the evaluation_data to the dataset
        raw_data = raw_data.append(user_data, ignore_index=True)

    data = feature_extraction(raw_data)

    # Keep all the traffic features

    # Split data into features and target
    features, target = split_input_and_target(data)

    # Normalise the columns
    normalised_features = normalise(features)

    # y = label_binariser(target)
    # binarised_targets = pandas.DataFrame(data=y)

    preprocessed_data = pandas.concat([normalised_features, target], axis=1, ignore_index=True)

    if user_data is not None:
        data = preprocessed_data.iloc[6000:, :].copy()
        features, target = split_input_and_target(data)
        return features
    else:
        preprocessed_data = balance_dataset(preprocessed_data)
        preprocessed_data.reset_index(inplace=True)
        return preprocessed_data


def label_binariser(target):
    lb = LabelBinarizer()
    lb.fit(target)
    target = lb.transform(target)
    return target

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

def balance_dataset(data):
    grouped_data = data.groupby(36, group_keys=False)
    data = grouped_data.apply(lambda x: x.sample(grouped_data.size().min()))
    return data

def feature_extraction(data):



    # Initial Analysis showed that any kind of feature selection results
    # in a reduction of accuracy in the logistic regression algorithm

    # Change these values to change the feature selection process
    # x, y = split_input_and_target(data)



    # Variance Threshold
    # selector = VarianceThreshold()
    # x = selector.fit_transform(x)

    # Select Precentile
    # selector = SelectPercentile(f_classif, percentile=100)
    # x = selector.fit_transform(x, y)

    # K Best
    # # x = SelectKBest(f_regression, k=36).fit_transform(x, y)

    # Recombine data
    # x = pandas.DataFrame(data=x[1:, 1:], columns=x[0, 1:])
    # data = pandas.concat([x, y], axis=1, ignore_index=True)
    # data = data.drop(data.index[len(data) - 1])
    return data




