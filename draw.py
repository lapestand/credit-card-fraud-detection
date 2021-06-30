import os
import sys
import time
import logging
import argparse
import pathlib

import properties
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    df = pd.read_csv(properties.DEFAULT_DATASET)
    print(df)