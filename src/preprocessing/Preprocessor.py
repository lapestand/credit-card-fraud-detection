import logging
import time

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
        self.mixed_transactions_path = os.path.join(script_path, "data", "mixed_transactions.csv")

        if not os.path.exists(self.quartiles_path):
            os.mkdir(self.quartiles_path)
            logging.debug(f"{self.quartiles_path} created")
        else:
            logging.debug(f"{self.quartiles_path} is exist therefore didn't created again")

        self.raw_data = pd.read_csv(self.raw_dataset_path)
        logging.info("Dataset loaded")

        self.pruned_data = raw_data[["Cardholder Last Name", "Cardholder First Initial", "Amount", "Vendor", "Transaction Date", "Merchant Category Code (MCC)"]
        logging.info("Dataset pruned")

        self.fraudulent_data = None
        
        self.pruned_fraudulent_data = None

        logging.debug(self.raw_data.columns)

        logging.info("Preprocessor created")


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

        logging.debug(f"Files created under {self.quartiles_path}")


    def get_percentage_of_quartiles(self, percentage):
        def percent(p, w):
            return (p * w) / 100.0

        arr, count = list(), list()
        for (root, dirs, files) in os.walk(self.quartiles_path):
            if not dirs:
                files = [os.path.join(root, file) for file in files]
                arr.append(random.sample(files, int(percent(percentage, len(files)))))
                count.append(len(files))

        logging.debug(f"Size of parts before selecting random {percentage}%: {count}")
        logging.debug(f"Size of parts after selecting random {percentage}%: {[len(_) for _ in arr]}")
        return arr


    def add_new_column(self, c_name, c_value=None):
        self.raw_data[c_name] = c_value
        logging.debug(f"New column '{c_name}' added with default value = {c_value}")


    def add_fake_instances(self, df_arr, group_keys):
        logging.debug("merging started")
        
        # merge all groups
        df = pd.concat([pd.read_csv(group) for quartile in df_arr for group in quartile])

        # get group names
        groups = [[f.split('_')[0], f.split('_')[-1][:-4]] for (r, d, f_l) in os.walk(self.quartiles_path) for f in f_l]


        # select random data in the merged dataset excluding current group

        mixed_transactions = pd.DataFrame(columns=list(df.columns))

        for idx, group" in enumerate(groups):
            
            # Get rows from df for current group
            current_group = df[(df[group_keys[0]] == group[0]) & (df[group_keys[1]] == group[1])]

            if len(df.index) > 0:
                # DF - group  -> difference
                df_without_current_group = df[(df[group_keys[0]] != group[0]) | (df[group_keys[1]] != group[1])]
                
                # Pick random samples from 'df_without_current_group' as large as the size of the current group
                fake_transactions = df_without_current_group.sample(n=len(current_group.index))

                # Set columns for fake transactions
                fake_transactions['Class'] = 'F'
                fake_transactions[group_keys[0]] = group[]
                fake_transactions[group_keys[1]] = group[]

                o_s = len(mixed_transactions.index)
                mixed_transactions = pd.concat([mixed_transactions, current_group, fake_transactions])
                n_s = len(mixed_transactions.index)
                c_s = len(current_group.index)
                f_s = len(fake_transactions)

                print(' '*150  , end='\r')
                print(f"Old size = {o_s}   fake_transactions = {f_s}   current_group = {c_s}  |  {n_s}   +{n_s-o_s}", end='\r')
        print('\n')
        mixed_transactions.to_csv(self.mixed_transactions_path)
        logging.debug(f"New dataset({mixed_transactions.shape}) saved into {self.mixed_transactions_path}")