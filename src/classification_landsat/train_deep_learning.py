import tensorflow as tf
from tensorflow import keras
from tensorflow import estimator
from sklearn.model_selection import train_test_split
import pickle
from sklearn.preprocessing import LabelBinarizer
import os
import numpy as np
import pandas


def train(data, analysis, hidden_layers, training=False, estimation=None, verbose=False):
    results = []
    dnn_clf = None

    data = data.iloc[:, 1:].copy()

    features, target = split_input_and_target(data)

    x_train, x_test, y_train, y_test = train_test_split(features, target, stratify=target, test_size=0.10)

    feature_cols = [tf.feature_column.numeric_column("X", shape=[1, 36])]


    dnn_clf = estimator.DNNClassifier(hidden_units=hidden_layers,
                                      n_classes=8,
                                      feature_columns=feature_cols)

    input_fn = estimator.inputs.numpy_input_fn(x={"X": x_train.values}, y=y_train.values, num_epochs=150, batch_size=10, shuffle=True)

    dnn_clf.train(input_fn=input_fn)

    test_input_fn = estimator.inputs.numpy_input_fn(
        x={"X": x_test.values}, y=y_test.values, shuffle=False)
    eval_results_testing = dnn_clf.evaluate(input_fn=test_input_fn)
    testing_results = [eval_results_testing['loss'], eval_results_testing['average_loss'], eval_results_testing['accuracy'], hidden_layers]

    train_input_fn = estimator.inputs.numpy_input_fn(
        x={"X": x_train.values}, y=y_train.values, shuffle=False)
    eval_results_training = dnn_clf.evaluate(input_fn=train_input_fn)
    training_results = [eval_results_training['loss'], eval_results_training['average_loss'], eval_results_training['accuracy'], hidden_layers]

    if verbose:
        print("The accuracy of training results is : {0:.2f}%".format(100 * eval_results_training['accuracy']))
        print("The accuracy of testing results is : {0:.2f}%".format(100 * testing_results['accuracy']))


    if analysis:
        results = testing_results + training_results
        return results

    if estimation is not None:

        # estimation["target"] = None

        input_fn = estimator.inputs.numpy_input_fn(x={"X": estimation.values}, y=None, shuffle=False)

        pred_generator = dnn_clf.predict(input_fn=input_fn)

        count = 0
        print("-"*90)
        indexs = []
        for prediction_instance in pred_generator:
            indexs.append(np.argmax(prediction_instance['probabilities']))

        return indexs

def split_input_and_target(data):
    """
    Split the feature and target columns
    :param data: The combined dataframe
    :return: features columns and target columns
    """

    features = data.iloc[:, :-1].copy()
    target = data.iloc[:, -1:].copy()

    return features, target
