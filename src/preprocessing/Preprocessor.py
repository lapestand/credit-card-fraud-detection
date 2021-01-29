"""
Preprocessor class for creating fraudulent data and new features from existing data 
"""

import logging
import time

import os
import pathlib
import shutil

import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer


class Preprocessor:
    def __init__(self, dataset_path=None):
        """[summary]

        Args:
            dataset_path ([type], optional): [description]. Defaults to None.

        Raises:
            TypeError: [description]
            FileNotFoundError: [description]
        """

        if not dataset_path:
            raise TypeError("Missing dataset path")
        
        if not os.path.isfile(dataset_path):
            raise FileNotFoundError(f"File not found -> {dataset_path}")
        
        self.raw_data_path  =   dataset_path
        
        self.repo_path      =   os.path.join("data", "repository")

        self.group_count    =   0


        if os.path.exists(self.repo_path):
            try:
                shutil.rmtree(self.repo_path)
            except Exception as e:
                logging.warning(f"{e}")
        
        os.mkdir(self.repo_path)
        logging.info(f"Repository folder created\t->\t{self.repo_path}")

        # Load the dataset
        try:
            self.raw_data = pd.read_csv(self.raw_data_path)
            logging.info("Dataset loaded")
        except IOError as e:
            logging.warning(f"{e}")
            raise

        self.necessary_features =   ["Cardholder Last Name", "Cardholder First Initial", "Amount", "Vendor", "Transaction Date", "Merchant Category Code (MCC)"]
        self.pruned_data        =   None

        self.group_labels       =   ['1st', '2nd', '3rd', '4th']
        self.groups             =   {el: [] for el in self.group_labels}

        logging.info("Preprocessor created")
    
    def preprocess(self, class_label, group_by, random_fraction_per_group, seed_val):
        
        # Prune unnecessary features from dataset
        self.pruned_data                    =   self.raw_data[self.necessary_features]
        logging.debug(f"Dataset pruned. Remaining columns are --> {', '.join(self.necessary_features)}")

        # Change the option to ignore false positive warning
        pd.set_option('chained_assignment',None)

        # Add class label to dataset
        self.pruned_data.loc[:, class_label[0]] = class_label[1]
        # logging.debug(f"New column '{class_label[0]}' added with default value = {class_label[1]}")

        # Group the dataset by given category list then return group count
        self.group_count                    =   self.split_by(group_by)

        # Get random sample from groups using random_fraction_per_group as fraction per group
        randomly_selected_groups            =   self.get_percentage_of_quartiles(random_fraction_per_group, seed_val)
        
        # Add fake transactions and merge groups
        self.add_fake_instances(randomly_selected_groups, group_by, seed_val)


    def create_random_seed_array(self, seed, _size, upper_bound):
        np.random.seed(seed)
        return np.random.randint(upper_bound, size=_size)

    
    def split_by(self, categories):
        group_count         =   dict.fromkeys(self.group_labels, 0)
        
        # create the groups using groupby
        groups              =   self.pruned_data.groupby(categories).size().reset_index(name='size')

        # determine the quartile values to use with pd.cut
        quartiles           =   groups['size'].quantile([.25, .5, .75]).tolist()

        # add a lower and upper range for the bins in pd.cut
        quartiles           =   [0] + quartiles + [float('inf')]

        # add a quartiles column to groups, using pd.cut
        groups['quartiles'] =   pd.cut(groups['size'], bins=quartiles, labels=self.group_labels)

        # merge df and groups
        df                  =   self.pruned_data.merge(groups, on=categories)

        # groupby on categories and quartiles
        dfg                 =   df.groupby(categories + ['quartiles'])

        # save the groups to self.groups
        for (fn, ln, q), g in dfg:
            self.groups[q].append({f"{fn}_{ln}": g.iloc[:, :-2]})
            # self.groups[q].append({[fn, ln]: g.iloc[:, :-2]})
            group_count[q]  +=  1

        # logging.debug(f"Files created under {self.quartiles_path}")
        logging.debug(f"Data splitted into groups by categories {categories}")
        
        return group_count

    
    def get_percentage_of_quartiles(self, percentage, seed_val):
        def percent(p, w):
            return (p * w) / 100.0
        
        # Create seed array using seed_val as seed
        random_elements_idx   =   {}
        
        # self.create_random_seed_array(seed=self.seed_val, _size=)
        for quartile in self.group_labels:
            random_elements_idx[quartile] =   self.create_random_seed_array(seed_val, int(percent(percentage*100, self.group_count[quartile])), self.group_count[quartile])
        
        
        #for (real), (new) in zip(self.group_count.items(), random_elements_idx.items()):
        #    print(f"""{real}\n{new[0], max(new[1])}""")

        randomly_selected_groups    =   []
        

        for (quartile, groups), (quartile_, idx) in zip(self.groups.items(), random_elements_idx.items()):
            # randomly_selected_groups.append({quartile: [groups[i] for i in idx]})
            randomly_selected_groups.append([groups[i] for i in idx])

        # print(list(randomly_selected_groups[-1]["4th"][-1].values())[0])
        return randomly_selected_groups


    def add_fake_instances(self, df_arr, group_labels, seed_val):
        logging.debug("Merging started")

        groups  =   [[list(group.keys()), list(group.values())] for quartile in df_arr for group in quartile]
        

        # merge all groups and reset indexes
        df      =   pd.concat([group[1][0] for group in groups]).reset_index(drop=True)

        # get group names
        groups  =   [[group[0][0].split('_') for group in groups]][0]

        print(df)
        print(df.columns)

        df = df.sample(frac=1, random_state=seed_val).reset_index(drop=True)
        print(df)
        print(df.columns)

        print(pd.DataFrame(groups, columns=['last name', "first initial"]))
        
        # select random data in the merged dataset excluding current group
        mixed_transactions = pd.DataFrame(columns=list(df.columns))

        for idx, group in enumerate(groups):
            # Get rows from df for current group
            current_group = df[(df[group_labels[0]] == group[0]) & (df[group_labels[1]] == group[1])]

            if len(df.index) > 0:
                # DF - group  -> difference
                df_without_current_group = df[(df[group_labels[0]] != group[0]) | (df[group_labels[1]] != group[1])]
                
                # Pick random samples from 'df_without_current_group' as large as the size of the current group
                fake_transactions = df_without_current_group.sample(n=len(current_group.index))

                # Set columns for fake transactions
                fake_transactions['Class'] = 'F'
                fake_transactions[group_labels[0]] = group[0]
                fake_transactions[group_labels[1]] = group[1]

                o_s = len(mixed_transactions.index)
                mixed_transactions = pd.concat([mixed_transactions, current_group, fake_transactions])
                n_s = len(mixed_transactions.index)
                c_s = len(current_group.index)
                f_s = len(fake_transactions)

                if idx % 10 == 0:
                    print(' '*150  , end='\r')
                    print(f"Old size = {o_s}   fake_transactions = {f_s}   current_group = {c_s}  |  {n_s}   +{n_s-o_s}", end='\r')
        print('\n')
        mixed_transactions.to_csv(self.mixed_transactions_path)
        logging.debug(f"New dataset({mixed_transactions.shape}) saved into {self.mixed_transactions_path}")
"""

class Preprocessor:
    def __init__(self, dataset_path=None, abs_repo_dir=None, script_path=None, repo_exist_ok=False, start_from=0, repo_name="data"):
        
        # Check if arguments passed else raise an error
        if (not dataset_path) or (not script_path):
            raise TypeError("Missing required positional argument")
        if not os.path.isfile(dataset_path):
            raise FileNotFoundError(f"File not found -> {dataset_path}")
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Folder not found -> {dataset_path}")

        # Start step for preprocessor
        self.start_from = start_from

        # Repository directory name
        self.repo_name = repo_name

        # Repository path
        self.repo_path = abs_repo_dir if abs_repo_dir else os.path.join(script_path, self.repo_name)

        # Callar script path 
        self.script_path = script_path


        # Raw dataset path
        self.raw_data_path = os.path.join(self.repo_path, "raw_data.csv")

        # Cleaned dataset path
        self.cleaned_raw_data_path = os.path.join(self.repo_path, "cleaned_raw_data.csv")

        # Groups path
        self.quartiles_path = os.path.join(self.repo_path, "groups")
        
        # Mixed transactions path
        self.mixed_transactions_path = os.path.join(self.repo_path, "mixed_transactions.csv")

        logging.info(f"Preprocessor repository directory\t->\t{self.repo_path}")

        if self.start_from == 0:
            if os.path.exists(self.repo_path):
                if not repo_exist_ok:
                    logging.warning(f"Existing directory for 'abs_repo_dir' -> {self.repo_path}")
                    raise FileExistsError(f"Existing directory for 'abs_repo_dir' -> {self.repo_path}")
                try:
                    shutil.rmtree(self.repo_path)
                except Exception as e:
                    logging.warning(f"{e}")
                    raise
            
            os.mkdir(self.repo_path)
            logging.info(f"Repository folder created\t->\t{self.repo_path}")

            shutil.copyfile(dataset_path, self.raw_data_path)

            logging.info(f"Repository folder created\t->\t{self.repo_path}")

        # Try to load given dataset
        try:
            self.raw_data = pd.read_csv(self.raw_data_path)
            logging.info("Dataset loaded")
        except IOError as e:
            logging.warning(f"{e}")
            raise

        self.cleaned_raw_data = None
        self.mixed_transactions = None

        if self.start_from <= 3:
            if not os.path.exists(self.quartiles_path):
                os.mkdir(self.quartiles_path)
                logging.debug(f"{self.quartiles_path} created")
            else:
                logging.debug(f"{self.quartiles_path} is exist therefore didn't created again")
        else:
            self.mixed_transactions = pd.read_csv(self.mixed_transactions_path)

        # Try to load given dataset
        try:
            self.raw_data = pd.read_csv(self.raw_data_path)
            logging.info("Dataset loaded")
        except IOError as e:
            logging.warning(f"{e}")
            raise

        logging.debug(self.raw_data.columns)

        logging.info("Preprocessor created")

    def preprocess():
        if self.start_from <= 2:
            self.cleaned_data = raw_data[["Cardholder Last Name", "Cardholder First Initial", "Amount", "Vendor", "Transaction Date", "Merchant Category Code (MCC)"]]
            logging.info("Dataset cleaned")
        else:
            self.cleaned_data = pd.read_csv(self.cleaned_raw_data_path)
            logging.info("Cleaned dataset loaded")


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

        for idx, group in enumerate(groups):
            
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
"""