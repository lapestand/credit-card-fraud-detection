import os
import pathlib

DESCRIPTION_MESSAGE = "This is my graduation project. It runs several classification algorithms to detect anomaly in " \
                      "credit card purchases."

C_ALGORITHMS = [ "LogisticRegression", "NaiveBayes", "K-NearestNeighbours", "DecisionTree", "RandomForest","StochasticGradientDescent", "SupportVectorMachine" ]

DATASET_NAME    =   "res_purchase_card_(pcard)_fiscal_year_2014_3pcd-aiuu.csv"

DEFAULT_DATASET = os.path.join(pathlib.Path(__file__).parent.absolute(), "data", DATASET_NAME)
