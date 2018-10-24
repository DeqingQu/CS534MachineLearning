import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv


def load_data(file_name, has_label=True):

    df = pd.read_csv(file_name, header=None)
    x = None
    y = None
    if has_label:
        y = df[[0]]
        x = df.drop([0], axis=1)

        def translate(v):
            return 1.0 if v == 3 else -1.0

        y = y.applymap(translate)
    else:
        x = df
    return x, y


if __name__ == '__main__':
    sample, label = load_data("pa2_train.csv")
    print(sample)
    print(label)
