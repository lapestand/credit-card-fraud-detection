import os
import sys
import time
import logging
import argparse
import pathlib

import pandas as pd
import matplotlib.pyplot as plt

import properties
from src.preprocessing.Preprocessor import Preprocessor
from src.modelling.Classifier import Classifier


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

    parser.add_argument("--algorithm", help="Test percentage", type=str)

    parser.add_argument("--bypass_preprocess", action="store_true")
    parser.add_argument("--apply_only_preprocess", action="store_true")
    


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


    if not args.bypass_preprocess:
        preprocessor = Preprocessor(dataset_path=properties.DEFAULT_DATASET)

        preprocessor.preprocess(
            class_label=["Class", 0],   #   0 FOR VALID 1 FOR FRADUENT
            group_by=['Cardholder Last Name', 'Cardholder First Initial'],
            random_fraction_per_group=0.5,
            seed_val=seed_val
            )
    else:
        logging.debug("Preprocesssing bypassed!")

    if not args.apply_only_preprocess:
        try:
            data    =   pd.read_csv(f"{seed_val}_generated.csv")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Couldn't find the dataset for the seed value ({args.seed})")

        classifier = Classifier(data, cross_validation=args.cross_validation, k=args.fold if args.fold else 10, test_ratio=args.test, algorithm=args.algorithm, seed_val=args.seed)

        logging.debug(f"Selected algorithm: {args.algorithm} via {'cross validation' if args.cross_validation else 'percentage split'}")

        classifier.analyze()
    else:
        logging.debug("apply_only_preprocess active")

    """
    logging.debug(f"Elapsed time for initialization            -> {it - s}")
    logging.debug(f"Elapsed time for class labeling            -> {ct - s}")
    logging.debug(f"Elapsed time for grouping                  -> {gt - s}")
    logging.debug(f"Elapsed time for random selection          -> {rt - s}")
    logging.debug(f"Elapsed time for fraud transation creation -> {ft - s}")
    logging.debug(f"TOTAL ELAPSED TIME                         -> {time.time() - s}")
    """
