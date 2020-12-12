import logging
import argparse

from properties import *
from src.preprocessing.Preprocessor import Preprocessor


def run():
    print("FRAUD DETECTED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=DESCRIPTION_MESSAGE)
    parser.add_argument("-A", "--algorithm", type=str,
                        help=f"Select classification algorithms.", choices=C_ALGORITHMS)

    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument("-V", "--version", help="show program version", action="store_true")
    parser.add_argument("-r", "--reset_model", help="reset model for selected algorithms", action="store_true")
    args = parser.parse_args()

    if args.version:
        print(VERSION)
    else:
        if args.verbose:
            VERBOSITY = True
            logging.basicConfig(level=logging.DEBUG)
        if args.reset_model:
            # reset_models()
            pass
