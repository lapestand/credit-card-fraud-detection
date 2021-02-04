import pandas as pd
import numpy as np

class NaiveBayesClassifier:
    def __init__(self, seed_val):
        self.seed_val   =   seed_val

    def train_test_split(self, data, class_label, test_size):
        train_set   =   data.sample(frac=(1.0 - test_size), random_state=self.seed_val)
        test_set    =   data.drop(train_set.index)

        train_x, train_y    =   train_set.loc[:, ~train_set.columns.isin([class_label])], train_set.loc[:, train_set.columns.isin([class_label])]
        test_x, test_y      =   test_set.loc[:, ~test_set.columns.isin([class_label])], test_set.loc[:, test_set.columns.isin([class_label])]
        
        return (train_x, train_y), (test_x, test_y)

    def classify(self, data=None, cross_validation=None, fold=None, test=None, class_label=None):
        if cross_validation:
            pass
        else:
            train_set, test_set =   self.train_test_split(data=data, class_label=class_label, test_size=test)

            print(train_set)

            print(test_set)
