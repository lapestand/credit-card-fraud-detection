import os
import time
import logging
import argparse
import pathlib
import properties

from src.preprocessing.Preprocessor import Preprocessor


def run():
    os.system("cls" if os.name == 'nt' else "clear")
    logging.info("Program started")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=properties.DESCRIPTION_MESSAGE)
    
    required_args = parser.add_argument_group('required arguments')
    
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    required_args.add_argument("-s", "--seed", help="Seed value for randomness", type=int, required=True)

    args = parser.parse_args()
    script_path = pathlib.Path(__file__).parent.absolute()

    
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

    preprocessor = Preprocessor(dataset_path=properties.DEFAULT_DATASET)

    preprocessor.preprocess(
        class_label=["Class", 'V'],
        group_by=['Cardholder Last Name', 'Cardholder First Initial'],
        random_fraction_per_group=0.5,
        seed_val=seed_val
        )
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
