import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer


class Preprocessor:
    def __init__(self, dataset_path=None, selected_algo=None):
        logging.info("Preprocessor creating")

        self.dateset_path = dataset_path
        self.raw_data = None
        logging.info("Preprocessor created")

    def __del__(self):
        pass

    def preprocess(self):
        logging.info("Dataset loading")
        self.raw_data = pd.read_csv(self.dateset_path)
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
