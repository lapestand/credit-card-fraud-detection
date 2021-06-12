import os
import time
import logging
import argparse

from properties import *
from src.preprocessing.PreprocessorOld import Preprocessor


class FraudDetector:
    def __init__(self, algorithms="ALL", reset=False, features="ALL"):
        logging.debug(f"test {pathlib.Path(__file__).parent.absolute()}")


def run():
    print("FRAUD DETECTED")


if __name__ == "__main__":
    os.system("cls" if os.name == 'nt' else "clear")
    parser = argparse.ArgumentParser(description=DESCRIPTION_MESSAGE)

    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    #    parser.add_argument("-d", "--dataset", help="Load dataset if not already loaded")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
        logging.info("VERBOSITY ON")
    else:
        logging.basicConfig(format='%(levelname)s :%(message)s', level=logging.WARNING)

    logging.info(f"SCRIPT PATH ==> {script_path}")
    logging.info(f"dataset ==> {DEFAULT_DATASET}")

    preprocessor_methods = [m for m in dir(Preprocessor) if not m.startswith('__')]

    elapsed_times = dict.fromkeys(preprocessor_methods, 0)
    status = dict.fromkeys(preprocessor_methods, False)
    

    preprocessor = Preprocessor(dataset_path=DEFAULT_DATASET, abs_repo_dir=script_path, script_path=script_path, repo_exist_ok=True)

    preprocessor.preprocess(
        class_label=["Class", 'V'],
        group_by=['Cardholder Last Name', 'Cardholder First Initial'],
        percantage_per_group=50,
        )

    pre_processor.add_new_column("Class", 'V')
    
    pre_processor.split_by(['Cardholder Last Name', 'Cardholder First Initial'])
    
    df_arr = pre_processor.get_percentage_of_quartiles(50)
    
    pre_processor.add_fake_instances(df_arr, ['Cardholder Last Name', 'Cardholder First Initial'])

    """
    logging.debug(f"Elapsed time for initialization            -> {it - s}")
    logging.debug(f"Elapsed time for class labeling            -> {ct - s}")
    logging.debug(f"Elapsed time for grouping                  -> {gt - s}")
    logging.debug(f"Elapsed time for random selection          -> {rt - s}")
    logging.debug(f"Elapsed time for fraud transation creation -> {ft - s}")
    logging.debug(f"TOTAL ELAPSED TIME                         -> {time.time() - s}")
    """
