import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import argparse
# import seaborn as sns
# import csv


def load_data(filename):
    data = np.genfromtxt(filename, dtype=np.str, delimiter=",")
    data = data.astype(float)
    # extract labels from data
    label = data[:, :1]
    # changes label 3 to 1, 5 to -1
    for i in label:
        if i[0] == 3:
            i[0] = 1
        else:
            i[0] = -1
    return data


def gini(data):
    # counts = class_counts(data) #a list that counts the number of all the possible labels
    # count3s = data[:, 0].count(1)
    # count5s = data[:, 0].count(-1)
    # counts = {1: count3s, -1: count5s}
    # print(data.shape)
    unique, count = np.unique(data, return_counts=True)
    counts = dict(zip(unique, count))
    # print(counts)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(data))
        impurity -= prob_of_lbl**2
    return impurity


if __name__ == '__main__':

    data_train = load_data("pa3_train_reduced.csv")
    data_valid = load_data("pa3_valid_reduced.csv")
    print(data_train[:2])
    print(data_valid[:2])

    # print(gini)
