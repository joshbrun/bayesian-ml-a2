# coding=utf-8

"""
SE755 - A2
Loads in the data from csv files

Authors:
Joshua Brundan
Kevin Hira
"""

import pandas
import os


def load(path, has_header):
    """
    Loads in the data from a csv file.
    :param path: The path to the data
    :param has_header: Has a header or not
    :return:
    """
    if has_header:
        header = 0
    else:
        header = None

    raw_data = pandas.read_csv(path, header=header)

    # print("Data:")
    # print(raw_data.describe())
    # print(path)

    return raw_data

def delete_file(path):
    """
    Remove a file
    :param path:
    :return:
    """
    if os.path.isfile(path):
        os.remove(path)

def append_to_file(path, line):
    """
    Appends a line to a file
    :param path:
    :param line:
    :return:
    """

    output = ",".join([str(x) for x in line])

    with open(path, 'a+') as f:
        f.write(output+"\n")
