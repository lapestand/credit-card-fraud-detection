import os
import time
import logging
import argparse

from properties import *
from src.preprocessing.Preprocessor import Preprocessor


class FraudDetector:
    def __init__(self, algorithms="ALL", reset=False, features="ALL"):
        logging.debug(f"test {pathlib.Path(__file__).parent.absolute()}")


def run():
    print("FRAUD DETECTED")


if __name__ == "__main__":
    os.system("cls" if os.name == 'nt' else "clear")
    parser = argparse.ArgumentParser(description=DESCRIPTION_MESSAGE)

    parser.add_argument("-V", "--version", help="show program version", action="store_true")
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    #    parser.add_argument("-d", "--dataset", help="Load dataset if not already loaded")

    args = parser.parse_args()
    script_path = pathlib.Path(__file__).parent.absolute()

    if args.version:
        print(VERSION)
    else:
        if args.verbose:
            logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
            logging.info("VERBOSITY ON")
        else:
            logging.basicConfig(format='%(levelname)s :%(message)s', level=logging.WARNING)

        logging.info(f"SCRIPT PATH ==> {script_path}")
        logging.info(f"dataset ==> {DEFAULT_DATASET}")


        s = time.time()

        pre_processor = Preprocessor(dataset_path=DEFAULT_DATASET, script_path=script_path)
        it = time.time()
        
        pre_processor.add_new_column("Class", 'V')
        ct = time.time()
        
        pre_processor.split_by(['Cardholder Last Name', 'Cardholder First Initial'])
        gt = time.time()
        
        df_arr = pre_processor.get_percentage_of_quartiles(50)
        rt = time.time()
        
        pre_processor.add_fake_instances(df_arr, ['Cardholder Last Name', 'Cardholder First Initial'])
        ft = time.time()

        logging.debug(f"Elapsed time for initialization            -> {it - s}")
        logging.debug(f"Elapsed time for class labeling            -> {ct - s}")
        logging.debug(f"Elapsed time for grouping                  -> {gt - s}")
        logging.debug(f"Elapsed time for random selection          -> {rt - s}")
        logging.debug(f"Elapsed time for fraud transation creation -> {ft - s}")
        logging.debug(f"TOTAL ELAPSED TIME                         -> {time.time() - s}")
