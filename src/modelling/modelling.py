class CrossValidationModel:
    def __init__(self) -> None:
        pass

def train_test_split(self, data, class_label, test_size, random_state):
        train_set   =   data.sample(frac=(1.0 - test_size), random_state=random_state)
        test_set    =   data.drop(train_set.index)

        train_x, train_y    =   train_set.loc[:, ~train_set.columns.isin([class_label])], train_set.loc[:, train_set.columns.isin([class_label])]
        test_x, test_y      =   test_set.loc[:, ~test_set.columns.isin([class_label])], test_set.loc[:, test_set.columns.isin([class_label])]
        
        return train_x, test_x, train_y, test_y