import logging

import os
import pathlib
from builtins import enumerate

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer


class Preprocessor:
    def __init__(self, dataset_path=None, script_path=None):
        self.raw_dataset_path = dataset_path
        self.splitted_data_path = os.path.join(script_path, "data", "splitted_by_cardholders")

        if not os.path.exists(self.splitted_data_path):
            os.mkdir(self.splitted_data_path)
            logging.debug(f"{self.splitted_data_path} created")
        else:
            logging.debug(f"{self.splitted_data_path} is exist therefore didn't created again")

        self.raw_data = pd.read_csv(self.raw_dataset_path)
        logging.debug("Dataset loaded")
        logging.debug(self.raw_data.columns)

        logging.info("Preprocessor created")

    def preprocess(self):
        logging.info("Dataset loading")
        self.raw_data = pd.read_csv(self.raw_dataset_path)
        logging.info("Dataset loaded")
        for idx, row in self.raw_data.iterrows():
            for col in self.raw_data.columns:
                if not row[col] or (row[col] and ((row[col].isspace()) if isinstance(row[col], str) else False)):
                    print(row[col], "EMPTY", col, end="\n\t")
                    print(f"row --> {row}")
        """
        logging.debug(f"Duplicates dropping. SHAPE {self.raw_data.shape}")
        pd.DataFrame.drop_duplicates(self.raw_data)
        logging.debug(f"Duplicates dropped. SHAPE {self.raw_data.shape}")

        logging.debug(f"Missing values handling")
        logging.info(
            "There isn't any missing value to handle in the dataset" if not self.raw_data.isnull().values.any() else "")
        logging.debug(f"Missing values handled")
        """

        """
                print(self.raw_data)
                print(self.raw_data[[self.raw_data.columns[0]]])
        """

    def split_by(self, categories):
        def split_(a, n):
            k, m = divmod(len(a), n)
            return [a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

        print(self.raw_data)
        groups = self.raw_data.groupby(list(categories))
        print(groups.first())
        exit(1)
        size_of_groups = self.raw_data.groupby(list(categories)).size()
        size_of_groups = size_of_groups.sort_values()
        logging.debug(f"Total group count: {len(groups)}")

        # logging.debug("DRAWING")
        # size_of_groups.plot(x="card holder", y="transaction count")
        # size_of_groups.sort_values().plot(x="card holder", y="transaction count")
        # print(size_of_groups.index)
        # print(size_of_groups.values)
        parts = split_(size_of_groups.values, 3)
        logging.debug(f"Partition count: {len(parts)}")
        # for idx, g_size in enumerate(size_of_groups):

        for idx, p in enumerate(parts):
            logging.debug(f"Size of part {idx} --> {len(p)}")
            logging.debug(f"\t{p[0]} - {p[-1]}")
#            for _ in p:


        # plt.setp(plt.axes().get_xticklabels(), visible=False)
        # plt.boxplot(size_of_groups[["transaction count"]])
        # plt.show()
