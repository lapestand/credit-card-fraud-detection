import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Preprocessor:
    def __init__(self, dataset_path=None, selected_algo=None):
        self.unprocessed_data = pd.read_csv("/data")
        logging.info("Preprocessor created")

    def __del__(self):
        pass
