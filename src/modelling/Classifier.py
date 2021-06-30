from numpy import mean, array_equal
from numpy.random import seed
from scipy.sparse import data
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

class Classifier:
    def __init__(self, dataset=None, cross_validation=False, k=10, test_ratio=0.3, algorithm=None, seed_val=None) -> None:
        self.dataset = dataset
        self.x = dataset.iloc[:, 0:len(dataset.columns) -1]
        self.y = dataset.iloc[:, len(dataset.columns) -1:]

        self.cross_validation = cross_validation
        self.k = k
        self.seed_val = seed_val

        if cross_validation:
            self.cv = KFold(n_splits=k, random_state=seed_val, shuffle=True)
        else:
            self.test_ratio = test_ratio
        
        self.models = {
            "LogisticRegression": LogisticRegression(random_state=self.seed_val),
            "K-NearestNeighbours": KNeighborsClassifier(n_neighbors=k, metric="minkowski"),
            "SupportVectorMachine": SVC(kernel='linear', random_state=seed_val),
            "GaussianNB": GaussianNB(),
            "DecisionTree-Entropy": DecisionTreeClassifier(criterion="entropy", random_state=seed_val),
            "DecisionTree-Gini": DecisionTreeClassifier(criterion="gini", random_state=seed_val),
            "RandomForest-Entropy": RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=seed_val),
            "RandomForest-Gini": RandomForestClassifier(n_estimators=10, criterion="gini", random_state=seed_val)
        }

        self.model = self.models[algorithm]
        self.analyze_result = {}

    """
    def train_test_split(self, data, class_label, test_size, seed_val):
        train_set   =   data.sample(frac=(1.0 - test_size), random_state=seed_val)
        test_set    =   data.drop(train_set.index)

        x_train, y_train    =   train_set.loc[:, ~train_set.columns.isin([class_label])], train_set.loc[:, train_set.columns.isin([class_label])]
        x_test, y_test      =   test_set.loc[:, ~test_set.columns.isin([class_label])], test_set.loc[:, test_set.columns.isin([class_label])]

        print(y_train)        
        return x_train, x_test, y_train, y_test
    """

    def print_result(self):
        print("\n\n\n\n\n\t\t\t\tRESULT\n\n")
        print(f"\n\nTrue positive: {self.analyze_result['tp']}\tFalse negative: {self.analyze_result['fn']}\tRecognition: {(self.analyze_result['tp'])/(self.analyze_result['tp'] + self.analyze_result['fn'])}")
        print(f"False positive: {self.analyze_result['fp']}\tTrue negative: {self.analyze_result['tn']}\tRecognition: {(self.analyze_result['tn'])/(self.analyze_result['fp'] + self.analyze_result['tn'])}")
        print(f"Total:\t{self.analyze_result['fp'] + self.analyze_result['tp'] + self.analyze_result['fn'] + self.analyze_result['tn']}\t\t\t\t\tRecognition:{(self.analyze_result['tp'] + self.analyze_result['tn'])/(self.analyze_result['tp'] + self.analyze_result['tn'] + self.analyze_result['fp'] + self.analyze_result['fn'])}\n\n")
        
        print("%")
        print(f"Accurracy:\t\t{self.analyze_result['accurracy']}")
        print(f"Error Rate:\t\t{100 - self.analyze_result['accurracy']}")
        print(f"Sensitivity:\t\t{self.analyze_result['sensitivity']}") # also known as TP rate OR Recall
        print(f"Specificity:\t\t{self.analyze_result['specificity']}")
        print(f"False Positive Rate:\t{100 - self.analyze_result['specificity']}")
        print(f"Precision:\t\t{self.analyze_result['precision']}")
        print(f"Prevalence:\t\t{self.analyze_result['prevalence']}")
        print(f"F1 Score:\t\t{self.analyze_result['f1_score']}")
    
    def cross_validate(self):
        X, y = shuffle(self.x, self.y, random_state=self.seed_val)
        # print(X, y)
        cm_list = []
        kf = KFold(self.k, shuffle=True, random_state=self.seed_val)
        for idx, (train_index, test_index) in enumerate(kf.split(self.x)):
            scaler = StandardScaler()
            # print("TRAIN:", train_index, "TEST:", test_index)
            # print(type(train_index))
            x_train, x_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]

            X_train = scaler.fit_transform(x_train)
            X_test = scaler.transform(x_test)
            self.model.fit(X_train, y_train.values.ravel())
            y_pred = self.model.predict(X_test)
            cm_list.append(confusion_matrix(y_test, y_pred))
            # print(cm_list[-1])
        # print("DONE")
        return mean(cm_list, axis=0)

    def train_test_split(self):
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=self.test_ratio, random_state=self.seed_val)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(x_train)
        X_test = scaler.transform(x_test)
        
        self.model.fit(X_train, y_train.values.ravel())
        y_pred = self.model.predict(X_test)

        # print(y_pred)
        # print(y_test.values.ravel())
        return confusion_matrix(y_test, y_pred)

    def analyze(self):
        if not self.cross_validation:
            cm = self.train_test_split()
        else:
            cm = self.cross_validate()
        
        tn, fn, tp, fp = cm[0][0], cm[1][0], cm[1][1], cm[0][1]
        
        self.analyze_result["tn"] = tn
        self.analyze_result["fn"] = fn
        self.analyze_result["tp"] = tp
        self.analyze_result["fp"] = fp
        self.analyze_result["sensitivity"] = (tp / (tp + fn)) * 100
        self.analyze_result["specificity"] = (tn / (tn + fp)) * 100
        self.analyze_result["precision"] = (tp / (tp + fp)) * 100
        self.analyze_result["prevalence"] = ((tp + fn) / ((tp + tn + fp + fn))) * 100
        self.analyze_result["accurracy"] = ((tp + tn) / (tp + tn + fp + fn)) * 100
        self.analyze_result["f1_score"] = ((2 * tp) / (2 * tp + fp + fn)) * 100
        
        self.print_result()

    
