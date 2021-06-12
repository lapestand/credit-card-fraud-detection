"""
Preprocessor class for creating fraudulent data and new features from existing data 
"""

import logging
import time
import datetime

import os
import pathlib
import shutil

import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder



class Preprocessor:
    """TODO
    
    Decrease fake instance count to frac(random(0.1, 0.5))
    """
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
        
        # self.repo_path      =   os.path.join("data", "repository")
        self.repo_path      =   os.path.join("workspace")

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

        self.derived_features   =   [
            "id",
            "Txn amount over month", "Average over 3 months", "Average daily over month", "Amount merchant type over month",
            "Number merchant type over month", "Amount merchant type over 3 months", "Amount same day", "Number same day",
            "Amount same merchant", "Number same merchant", "Amount merchant over 3 months", "Number merchant over 3 months",
            "Class"
            ]
        logging.info("Preprocessor created")


    def preprocess(self, class_label, group_by, random_fraction_per_group, seed_val):
        
        start   =   time.time()
        
        # Change the option to ignore false positive warning
        pd.set_option('chained_assignment', None)
        logging.info("chained_assignment option of pd is closed due to avoid false positive")

        # Prune unnecessary features from dataset
        self.pruned_data                                        =   self.raw_data[self.necessary_features]
        logging.debug(f"Dataset pruned. Remaining columns are --> {', '.join(self.necessary_features)}")


        # print(self.pruned_data[["Vendor", "Merchant Category Code (MCC)"]])
        # Convert "Merchant type" and "Vendor" attributes
        label_encoder                                           =   LabelEncoder()
        
        label_encoder.fit(self.pruned_data['Merchant Category Code (MCC)'])
        self.pruned_data['Merchant Category Code (MCC)']        =   label_encoder.transform(self.pruned_data['Merchant Category Code (MCC)'])

        logging.debug("Merchant Category Code (MCC) label encoded")

        label_encoder.fit(self.pruned_data['Vendor'])
        self.pruned_data['Vendor']                              =   label_encoder.transform(self.pruned_data['Vendor'])

        logging.debug("Vendor label encoded")


        # print(self.pruned_data[["Vendor", "Merchant Category Code (MCC)"]])

        # Add class label to dataset
        self.pruned_data.loc[:, class_label[0]] = class_label[1]
        logging.debug(f"New column '{class_label[0]}' added with default value = {class_label[1]}")

        # Group the dataset by given category list then return group count
        self.group_count                    =   self.split_by(group_by)

        # Get random sample from groups using random_fraction_per_group as fraction per group
        randomly_selected_groups            =   self.get_percentage_of_quartiles(random_fraction_per_group, seed_val)
        
        # Add fake transactions and merge groups
        fraduent_transactions               =   self.add_fake_instances(randomly_selected_groups, group_by, seed_val)
        # fraduent_transactions               =   self.add_fake_instances(randomly_selected_groups, group_by, seed_val).to_csv(f"faked_seed_{seed_val}.csv", index=False)
        # self.new_add_fake_instances(randomly_selected_groups, group_by, seed_val)

        # Create new features
        # fraduent_transactions               =   pd.read_csv(f"faked_seed_{seed_val}.csv")
        self.derived_data                     =   self.derive_features(fraduent_transactions, class_label, group_by, seed_val)
        self.derived_data.to_csv(f"{seed_val}_generated.csv", index=False)
        logging.debug("New dataset created from derived features")
        
        

        logging.info(f"Elapsed time as second: {time.time() - start}")


    def derive_features(self, old_df, class_label, group_by, seed_val):
        logging.debug("Feature derivation started")

        old_df["Transaction Date"]  =   pd.to_datetime(old_df["Transaction Date"], format="%m/%d/%Y %I:%M:%S %p")

        # print(old_df[["Merchant Category Code (MCC)", "Vendor"]])

        # merchants                   = old_df.groupby("Merchant Category Code (MCC)")

        # print(old_df.dtypes)

        ONE_MONTH                   =   30
        WEEK_COUNT_IN_ONE_MONTH     =   4

        cards                       =   old_df.groupby(group_by)
        total_transaction_count     =   len(old_df.index)
        current                     =   0
        current_group               =   0

        # new_transactions    =   pd.DataFrame(columns=self.derived_features)

        new_transactions    =   {el: [] for el in self.derived_features}
        
        for (card_holder_last_name, card_holder_first_initial), transactions in cards:
            # print(f"Card holder {card_holder_first_initial} {card_holder_last_name} transaction count: {len(transactions.index)}")

            # TRANSACTION_COUNT_OF_CURRENT_CARD   =   len(transactions.index)

            for _, transaction in transactions.iterrows():
                transactions_until_current_transaction  =   transactions.loc[transactions["Transaction Date"] <= transaction["Transaction Date"]]
                
                # last_month_mask = (transactions["Transaction Date"] > transaction["Transaction Date"] - datetime.timedelta(30)) & (transactions["Transaction Date"] <= transaction["Transaction Date"])

                last_3_month_mask               =   transactions_until_current_transaction["Transaction Date"] > transaction["Transaction Date"] - datetime.timedelta(ONE_MONTH * 3)
                transactions_in_last_3_month    =   transactions_until_current_transaction.loc[last_3_month_mask]

                last_month_mask                 =   transactions_in_last_3_month["Transaction Date"] > transaction["Transaction Date"] - datetime.timedelta(ONE_MONTH)
                transactions_in_last_month      =   transactions_in_last_3_month.loc[last_month_mask]

                same_day_mask                   =   transactions_in_last_month["Transaction Date"] == transaction["Transaction Date"]
                transactions_in_same_day        =   transactions_in_last_month.loc[same_day_mask]
                
                # Group ID
                new_transactions["Id"].append(current_group)

                # Txn amount over month
                new_transactions["Txn amount over month"].append(transactions_in_last_month.loc[:, "Amount"].sum() / len(transactions_in_last_month.index))

                # Average over 3 months
                new_transactions["Average over 3 months"].append(transactions_in_last_3_month.loc[:, "Amount"].sum() / (WEEK_COUNT_IN_ONE_MONTH * 3))

                # Average daily over month
                new_transactions["Average daily over month"].append(transactions_in_last_month.loc[:, "Amount"].sum() / ONE_MONTH)

                # Amount merchant type over month
                same_merchant_type_over_month_mask          =   transactions_in_last_month["Merchant Category Code (MCC)"] == transaction["Merchant Category Code (MCC)"]
                new_transactions["Amount merchant type over month"].append(transactions_in_last_month.loc[same_merchant_type_over_month_mask, "Amount"].sum() / ONE_MONTH)

                # Number merchant type over month
                new_transactions["Number merchant type over month"].append(len(transactions_in_last_month.loc[same_merchant_type_over_month_mask].index))

                # Amount merchant type over 3 months
                same_merchant_over_3_month_mask   =   transactions_in_last_3_month["Merchant Category Code (MCC)"] == transaction["Merchant Category Code (MCC)"]
                new_transactions["Amount merchant type over 3 months"].append(transactions_in_last_3_month.loc[same_merchant_over_3_month_mask, "Amount"].sum() / (WEEK_COUNT_IN_ONE_MONTH * 3))

                # Amount same day
                new_transactions["Amount same day"].append(transactions_in_same_day.loc[:, "Amount"].sum())

                # Number same day
                new_transactions["Number same day"].append(len(transactions_in_same_day.index))

                # Amount same merchant
                same_merchant_mask                      =   transactions_in_same_day["Vendor"] == transaction["Vendor"]
                new_transactions["Amount same merchant"].append(transactions_in_same_day.loc[same_merchant_mask, "Amount"].sum())

                # Number same merchant
                new_transactions["Number same merchant"].append(len(transactions_in_same_day.loc[same_merchant_mask].index))

                # Amount merchant over 3 months
                same_merchant_over_3_months_mask        =   transactions_in_last_3_month["Vendor"] == transaction["Vendor"]
                new_transactions["Amount merchant over 3 months"].append(transactions_in_last_3_month.loc[same_merchant_over_3_months_mask, "Amount"].sum() / (WEEK_COUNT_IN_ONE_MONTH * 3))

                # Number merchant over 3 months
                new_transactions["Number merchant over 3 months"].append(len(transactions_in_last_3_month.loc[same_merchant_over_3_months_mask].index))

                # Class
                new_transactions["Class"].append(transaction["Class"])


                current += 1
                print("                                              "*3, end='\r')
                print(f"Total transaction: {total_transaction_count} | Current transaction: {current} && Progress %{(current / total_transaction_count) * 100}", end='\r')
            current_group += 1
        print()
        return pd.DataFrame.from_dict(new_transactions)


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
            # print(f"{fn}_{ln}")
            self.groups[q].append({f"{fn}_{ln}": g.iloc[:, :-2]})
            # print(g.iloc[:, :-2])
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
        logging.debug("FAKE INSTANCE GENERATING STARTED")

        # Extract groups from dict
        groups  =   [[list(group.keys()), list(group.values())] for quartile in df_arr for group in quartile]
        

        # Merge all groups and reset indexes
        df      =   pd.concat([group[1][0] for group in groups]).reset_index(drop=True)

        logging.debug(f"Shape of given dataset before generation: {df.shape}")

        # Get group names
        groups  =   [[group[0][0].split('_') for group in groups]][0]
        
        # Shuffle merged group using given seed to increase randomness
        df = df.sample(frac=1, random_state=seed_val).reset_index(drop=True)

        # print(pd.DataFrame(groups, columns=['last name', "first initial"]))
        
        # Select random data in the merged dataset excluding current group
        # mixed_transactions = pd.DataFrame(columns=list(df.columns))

        total_len = len(groups)

        # Create empty dataframe using 
        mixed_transactions = pd.DataFrame(columns=list(df.columns))
        mixed_transactions  =   []  

        for idx, group in enumerate(groups):
            # Get rows from df for current group
            
            # s = time.time()
            current_group = df[(df[group_labels[0]] == group[0]) & (df[group_labels[1]] == group[1])]
            # print(f"current_group = df[(df[group_labels[0]] == group[0]) & (df[group_labels[1]] == group[1])] time is {time.time() - s}")
            if len(df.index) > 0:
                # DF - group  -> difference
                # s = time.time()
                df_without_current_group = df[(df[group_labels[0]] != group[0]) | (df[group_labels[1]] != group[1])]
                # print(f"df_without_current_group = df[(df[group_labels[0]] != group[0]) | (df[group_labels[1]] != group[1])] time is {time.time() - s}")
                
                # Pick random samples from 'df_without_current_group' as large as the size of the current group
                # s = time.time()
                fake_transactions = df_without_current_group.sample(n=len(current_group.index), random_state=seed_val)
                # print(f"fake_transactions = df_without_current_group.sample(n=len(current_group.index), random_state=seed_val) time is {time.time() - s}")
                
                # Set columns for fake transactions
                # s = time.time()
                fake_transactions['Class'] = 1   #   0 FOR VALID 1 FOR FRADUENT
                fake_transactions[group_labels[0]] = group[0]
                fake_transactions[group_labels[1]] = group[1]
                # print(f"FAKE  time is {time.time() - s}")

                # s = time.time()
                mixed_transactions.extend([current_group, fake_transactions])
                # print(f"Concat time is {time.time() - s}")
                
                # print("\n\n\n")

                print(' '*150  , end='\r')
                print(f"Group count = {total_len}\tCurrent group = {idx + 1}\tRemained = {total_len - idx - 1} ||| Fake Transaction count = {len(fake_transactions.index)}", end='\r')
        print(' '*150  , end='\r')

        mixed_transactions = pd.concat(mixed_transactions)

        mixed_transactions.drop_duplicates(keep="first", ignore_index=True, inplace=True)

        mixed_transactions  =   mixed_transactions.sample(n=len(mixed_transactions.index), random_state=seed_val)

        mixed_transactions.reset_index(drop=True, inplace=True)

        logging.debug(f"New dataset({mixed_transactions.shape})")
        return mixed_transactions
        # logging.debug(f"New dataset({mixed_transactions.shape}) saved into {self.mixed_transactions_path}")

    # IN PROGRESS
    def new_add_fake_instances(self, df_arr, group_labels, seed_val):
        logging.debug("FAKE INSTANCE GENERATING STARTED")

        # Extract groups from dict
        groups  =   [[list(group.keys()), list(group.values())] for quartile in df_arr for group in quartile]

        # Merge all groups and reset indexes
        df      =   pd.concat([group[1][0] for group in groups]).reset_index(drop=True)

        logging.debug(f"Shape of given dataset before generation: {df.shape}")

        # Get group names
        groups  =   [[group[0][0].split('_') for group in groups]][0]


        # Shuffle merged group using given seed to increase randomness
        df = df.sample(frac=1, random_state=seed_val).reset_index(drop=True)

        new_groups          =   df.groupby(group_labels)
        indexes             =   self.create_random_seed_array(seed_val, len(new_groups), len(new_groups))

        for real_idx, idx in enumerate(indexes):
            if real_idx == idx:
                if idx == len(indexes):
                    indexes[real_idx] -= 1
                else:
                    indexes[real_idx] += 1
        
        
        
        for idx, (group_name, group_content) in enumerate(new_groups):
            if len(df.index) > 0:
                
                
                print(f"{idx}\t->\t{group_name}\t->{len(group_content.index)}")
            
        print(len(new_groups))



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


"""
df.groupby(['Cardholder Last Name', 'Cardholder First Initial', pd.Grouper(key='Transaction Date', freq="30D")]).agg({'Cardholder Last Name': 'first', 'Cardholder First Initial': 'first', 'Amount': lambda x: sum(x) / 30})
"""