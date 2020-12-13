import logging
import argparse

from properties import *
from src.preprocessing.Preprocessor import Preprocessor


class FraudDetector:
    def __init__(self, algorithms="ALL", reset=False, features="ALL"):
        pass


def run():
    print("FRAUD DETECTED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=DESCRIPTION_MESSAGE)

    parser.add_argument("-V", "--version", help="show program version", action="store_true")
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument("-r", "--reset_model", help="reset model for selected algorithms", action="store_true")
    #    parser.add_argument("-d", "--dataset", help="Load dataset if not already loaded")

    required = parser.add_argument_group("required arguments")
    required.add_argument("-A", "--algorithm", nargs='+', help="Select classification algorithms.",
                          choices=C_ALGORITHMS, required=True)

    """required.add_argument("-rF", "--remove_feature", type=str, help="Remove feature from dataset.",
                          choices=[], required=True)"""

    args = parser.parse_args()

    if args.version:
        print(VERSION)
    else:
        if args.verbose:
            logging.basicConfig(level=logging.DEBUG)
        if 'ALL' in args.algorithm:
            args.algorithm = ['ALL']
        pre_processor = Preprocessor(dataset_path=DEFAULT_DATESET, selected_algo=args.algorithm)
