import os
import sys
import time
import logging
import argparse
import pathlib
import properties

import pandas as pd

from src.preprocessing.Preprocessor import Preprocessor

from src.modelling.algorithms.NaiveBayes import NaiveBayesClassifier


def run():
    os.system("cls" if os.name == 'nt' else "clear")
    logging.info("Program started")


if __name__ == "__main__":
    os.system("cls" if os.name == 'nt' else "clear")
    parser = argparse.ArgumentParser(description=properties.DESCRIPTION_MESSAGE)
    
    required_args = parser.add_argument_group('required arguments')
    
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    required_args.add_argument("-s", "--seed", help="Seed value for randomness", type=int, required=True)
    
    parser.add_argument("--split_percentage", help="Validation method for classifier", action="store_true")
    parser.add_argument("--test", help="Test percentage", type=float)

    parser.add_argument("--cross_validation", help="Validation method for classifier", action="store_true")    
    required_args.add_argument("--fold", help="Fold count", type=int)


    args = parser.parse_args()
    script_path = pathlib.Path(__file__).parent.absolute()

    if (not args.cross_validation) and (not args.split_percentage):
        raise AttributeError("Cross validation or split percentage should be selected")

    if args.cross_validation:
        if not args.fold:
            raise AttributeError("Fold is necessarry if Cross validation selected")
    
    
    if args.split_percentage:
        if not args.test:
            raise AttributeError("Test is necessarry if Split percentage selected")


    
    if args.verbose:
        logging.basicConfig(format='%(levelname)s:\t%(message)s', level=logging.DEBUG)
        logging.info("VERBOSITY ON")
    else:
        logging.basicConfig(format='%(levelname)s :\t%(message)s', level=logging.WARNING)
    
    seed_val = args.seed

    logging.info(f"Seed value is {seed_val}")

    logging.info(f"SCRIPT PATH ==> {script_path}")
    logging.info(f"dataset ==> {properties.DEFAULT_DATASET}")

    """
    preprocessor_methods = [m for m in dir(Preprocessor) if not m.startswith('__')]

    elapsed_times = dict.fromkeys(preprocessor_methods, 0)
    status = dict.fromkeys(preprocessor_methods, False)
    """

    # preprocessor = Preprocessor(dataset_path=properties.DEFAULT_DATASET, abs_repo_dir=script_path, script_path=script_path, repo_exist_ok=True)

    if False:
        preprocessor = Preprocessor(dataset_path=properties.DEFAULT_DATASET)

        preprocessor.preprocess(
            class_label=["Class", 0],   #   0 FOR VALID 1 FOR FRADUENT
            group_by=['Cardholder Last Name', 'Cardholder First Initial'],
            random_fraction_per_group=0.5,
            seed_val=seed_val
            )

    data    =   pd.read_csv(f"{seed_val}_generated.csv")

    naive_bayes_classifier  =   NaiveBayesClassifier(seed_val=seed_val)
    if args.split_percentage:
        naive_bayes_classifier.classify(data=data, cross_validation=False, test=args.test, class_label="Class")
    else:
        naive_bayes_classifier.classify(data=data, cross_validation=True, fold=args.fold, class_label="Class")

    """
    preprocessor.add_new_column("Class", 'V')
    
    preprocessor.split_by(['Cardholder Last Name', 'Cardholder First Initial'])
    
    df_arr = preprocessor.get_percentage_of_quartiles(50)
    
    preprocessor.add_fake_instances(df_arr, ['Cardholder Last Name', 'Cardholder First Initial'])
    """
    """
    logging.debug(f"Elapsed time for initialization            -> {it - s}")
    logging.debug(f"Elapsed time for class labeling            -> {ct - s}")
    logging.debug(f"Elapsed time for grouping                  -> {gt - s}")
    logging.debug(f"Elapsed time for random selection          -> {rt - s}")
    logging.debug(f"Elapsed time for fraud transation creation -> {ft - s}")
    logging.debug(f"TOTAL ELAPSED TIME                         -> {time.time() - s}")
    """
