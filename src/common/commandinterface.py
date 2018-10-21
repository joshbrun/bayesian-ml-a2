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
        print("Execution Mode:")
        self.options = {'train': False,
                        'dataset': None,
                        'newdata': None}
        parser = argparse.ArgumentParser()

        parser.add_argument("-d", "--dataset", type=int, choices=[1, 2, 3, 4, 5],
                            help="Pick a particular data set, options [1:regression, 2:cluster, 3: Classification]")
        parser.add_argument("-s", "--select", type=int, choices=[0, 1, 2],
                            help="Select the algorithm to run, options [0:Both, 1:first, 2:second], for regression: has no effect only has Bayesian linear regression, for classification: first: logistic_regression, second: deep learning and for clustering first: k_means, second: gaussian_mixture_model")

        parser.add_argument("-n", "--newdata", help="Process a new data instance and perform an estimation")
        parser.add_argument("-t", "--train", help="If present train the models", action="store_true")
        parser.add_argument("-a", "--analysis", help="If present run the analysis of the models, The analysis flag overrides other flags", action="store_true")

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
        if not self.args.analysis and not self.args.train and not (self.args.dataset or self.args.newdata):
            print("\nError: No options specified.")
            print("\tRead the README File  or")
            print("\tRun the command:")
            print("\t\tpython run.py --help\n")
            self.options = None

        dataset = self.args.dataset
        dataset_dict = {1: 'Regression', 2: 'Clustering', 3: 'Classification'}


        if dataset == 1:
            print("\tSingle data set: %d: %s" % (dataset, dataset_dict[dataset]))

        elif dataset == 2:
            print("\tSingle data set: %d: %s" % (dataset, dataset_dict[dataset]))
            if self.args.select == 0:
                dataset = [2, 3]
                print("\tBoth algorithms selected k-means and GMM")
            elif self.args.select == 1:
                datase = [2]
                print("\tSingle algorithms selected k-means")
            elif self.args.select == 2:
                dataset = [3]
                print("\tSingle algorithms selected GMM")

        elif dataset == 3:
            print("\tSingle data set: %d: %s" % (dataset, dataset_dict[dataset]))
            if self.args.select == 0:
                dataset = [4, 5]

                print("\tBoth algorithms selected logistic_regression and deep_learning")
            elif self.args.select == 1:
                dataset = [4]

                print("\tSingle algorithms selected logistic_regression")
            elif self.args.select == 2:
                dataset = [5]

                print("\tSingle algorithms selected deep_learning")

        else:
            dataset = [1, 2, 3, 4, 5]
            print("\tAll data sets:")
            print("\t\t1: Regression")
            print("\t\t2: Clustering")
            print("\t\t\t2.1: K-means")
            print("\t\t\t2.2: GMM")
            print("\t\t3: Classification")
            print("\t\t\t3.1: logistic regression")
            print("\t\t\t3.2: Deep-learning")


        if not self.args.analysis:
            print("\tTraining: %s" % self.args.train)



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
                if not self.args.analysis:
                    print("\tPrediction on rows in:")
                    print("\t\tFile: %s" % self.args.newdata)

            if type(dataset) is not list:
                dataset = [dataset]


        if self.options is not None:
            self.options = {'train': self.args.train,
                            'dataset': dataset,
                            'newdata': self.args.newdata,
                            'analysis': False}

        if self.args.analysis:
            self.options = {'train': True,
                            'dataset': dataset,
                            'newdata': None,
                            'analysis': True}
            print("\tAnalysis Mode Only.")

    def get_options(self):
        """
        Getter for the verified options
        The format of the options is a dictionary with indexs train, dataset and newdata.
        :return: options
        """
        return self.options
