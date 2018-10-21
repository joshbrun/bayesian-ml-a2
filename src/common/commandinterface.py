# coding=utf-8

"""
SE755 - A2s
The Command Line interface which serves as an interface for the machine learning training and estimation.

Authors:
Joshua Brundan
Kevin Hira
"""

import argparse


class CommandInterface:
    """
    The Command interface serving as the user interface for this machine learning project
    Allows a user to train against the various methods, each associated with the corresponding dataset.
    Also allows the user to input a new file to be estimated against.
    """

    def __init__(self):
        """
        Initialises the command interface, extracts the arguments, verifies them, extracts options from them
        """

        self.options = {'train': False,
                        'dataset': None,
                        'newdata': None}
        parser = argparse.ArgumentParser()

        parser.add_argument("-d", "--dataset", type=int, choices=[1, 2, 3],
                            help="Pick a particular data set, options [1:regression, 2:cluster, 3: Classification]")

        parser.add_argument("-n", "--newdata", help="Process a new data instance and perform an estimation")
        parser.add_argument("-t", "--train", help="If present train the models", action="store_true")
        self.args = parser.parse_args()
        self.verify_args()

    def verify_args(self):
        """
        Verify the correctness of the arguments.
        Rules:
        1: can only estimate new data on a single dataset.
        therefore for -n argument to be present the -d must also be present.
        2: must have arguments, default behaviour is to exit with a meaningful message
        3: can always train
        """

        # Check there is some argument, [Rule 2 and Rule 3]
        if not self.args.train and not (self.args.dataset or self.args.newdata):
            print("\nError: No options specified.")
            print("\tRead the README File  or")
            print("\tRun the command:")
            print("\t\tpython run.py --help\n")
            self.options = None

        dataset = self.args.dataset
        dataset_dict = {1: 'Regression', 2: 'Clustering', 3: 'Classification'}

        # Check if all or a single dataset
        if dataset in [1, 2, 3]:
            print("\nRunning against dataset:")
            print("\t%d: %s" % (dataset, dataset_dict[dataset]))
        else:
            dataset = [1, 2, 3]
            print("\nDatasets running against:")
            print("\t1: Regression")
            print("\t2: Clustering")
            print("\t3: Classification")

        print("\nTraining: %s" % self.args.train)

        # Check if newdata is only called on a single dataset [Rule 1]
        if dataset == [1, 2, 3]:
            if self.args.newdata:
                self.options = None
                print("\nOptions Error: estimation of newdata can only be done against one type.")
                print("\tPlease use the  -d N  argument to specify exactly one dataset that corresponds with the newdata")
                print("\nFor more information:")
                print("\tRead the README File  or")
                print("\tRun the command:")
                print("\t\tpython run.py --help\n")
        else:
            if self.args.newdata:
                print("\n%s on new data set:" % dataset_dict[dataset])
                print("\tFile: %s" % self.args.newdata)

        if self.options is not None:
            self.options = {'train': self.args.train,
                            'dataset': dataset,
                            'newdata': self.args.newdata}

    def get_options(self):
        """
        Getter for the verified options
        The format of the options is a dictionary with indexs train, dataset and newdata.
        :return: options
        """
        return self.options
