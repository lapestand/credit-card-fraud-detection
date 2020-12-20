import logging

import os
import pathlib

import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer


class Preprocessor:
    def __init__(self, dataset_path=None, script_path=None):
        self.raw_dataset_path = dataset_path
        self.script_path = script_path
        self.quartiles_path = os.path.join(script_path, "data", "groups")

        if not os.path.exists(self.quartiles_path):
            os.mkdir(self.quartiles_path)
            logging.debug(f"{self.quartiles_path} created")
        else:
            logging.debug(f"{self.quartiles_path} is exist therefore didn't created again")

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
        # create the groups using groupby
        groups = self.raw_data.groupby(categories).size().reset_index(name='size')

        # determine the quartile values to use with pd.cut
        quartiles = groups['size'].quantile([.25, .5, .75]).tolist()

        # add a lower and upper range for the bins in pd.cut
        quartiles = [0] + quartiles + [float('inf')]

        # add a quartiles column to groups, using pd.cut
        groups['quartiles'] = pd.cut(groups['size'], bins=quartiles, labels=['1st', '2nd', '3rd', '4th'])

        # merge df and groups
        df = self.raw_data.merge(groups, on=categories)

        # groupby on categories and quartiles
        dfg = df.groupby(categories + ['quartiles'])

        # save the groups to individual csv files
        for (fn, ln, q), g in dfg:
            # create the path
            path = pathlib.Path(f'{self.quartiles_path}/{q}')

            # make the directory
            path.mkdir(parents=True, exist_ok=True)

            # write the file without the size and quartiles columns
            g.iloc[:, :-2].to_csv(path / f'{fn}_{ln}.csv', index=False)

    def get_percentage_of_quartiles(self, percentage):
        def percent(p, w):
            return (p * w) / 100.0

        arr, tmp = list(), list()
        for (root, dirs, files) in os.walk(self.quartiles_path):
            if not dirs:
                files = [os.path.join(root, file) for file in files]
                arr.append(random.sample(files, int(percent(percentage, len(files)))))
                tmp.append(range(len(files)))

        logging.debug(f"Size of parts before selecting random {percentage}%: {[len(_) for _ in tmp]}")
        logging.debug(f"Size of parts after selecting random {percentage}%: {[len(_) for _ in arr]}")
