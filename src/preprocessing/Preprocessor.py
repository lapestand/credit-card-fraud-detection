import logging

import os
import pathlib
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
        groups = self.raw_data.groupby(list(categories))
        size_of_groups = self.raw_data.groupby(list(categories)).size()
        print(size_of_groups)
        print(f"Total group count: {len(groups)}", end="\n\n\n")
        logging.debug("DRAWING")
        size_of_groups.plot(x="card holder", y="transaction count")
        plt.setp(plt.axes().get_xticklabels(), visible=False)
        # plt.boxplot(size_of_groups[["transaction count"]])
        plt.show()
        print("\n")
        print("splitted")
