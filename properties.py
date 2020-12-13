import os

DESCRIPTION_MESSAGE = "This is my graduation project. It runs several classification algorithms to detect anomaly in " \
                      "credit card purchases."

C_ALGORITHMS = ["ALL", "LogisticRegression", "NaiveBayes", "StochasticGradientDescent", "K-NearestNeighbours",
                "DecisionTree", "RandomForest", "SupportVectorMachine"]

VERSION = "CCFD 1.0.1"

DEFAULT_DATESET = os.path.join("data", "res_purchase_card_(pcard)_fiscal_year_2014_3pcd-aiuu.csv")
