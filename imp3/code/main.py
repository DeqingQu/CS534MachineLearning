import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import argparse
# import seaborn as sns
# import csv


def load_data(file_name, has_label=True):

    df = pd.read_csv(file_name, header=None)
    y = None
    if has_label:
        y = df[[0]]
        x = df.drop([0], axis=1)

        #   if v == 3, y = 1.0; if v == 5, y = -1.0
        def translate(v):
            return 1.0 if v == 3 else -1.0
        y = y.applymap(translate)

        #   translate to np.array
        x = np.array(x)
        y = np.array(y)

        #   insert bias
        x = np.insert(x, 0, values=1.0, axis=1)

        #   transfer label to array, not matrix
        y = y.T[0]
    else:
        x = df
        #   translate to np.array
        x = np.array(x)

        #   insert bias
        x = np.insert(x, 0, values=1.0, axis=1)
    return x, y


if __name__ == '__main__':

    sample_train, label_train = load_data("pa3_train_reduced.csv")
    sample_valid, label_valid = load_data("pa3_valid_reduced.csv")
    print(sample_train[:2])
    print(label_train[:2])
    print(sample_valid[:2])
    print(label_valid[:2])
