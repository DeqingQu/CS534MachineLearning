import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv


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
    else:
        x = df
        #   translate to np.array
        x = np.array(x)
    return x, y


def online_training(x_train, y_train, x_valid, y_valid, iters=15):
    w = np.zeros(len(x_train[0]))
    for i in range(iters):
        for t in range(len(x_train)):
            u = np.sign(x_train[t].dot(w))
            if u * y_train[t] <= 0:
                w += y_train[t] * x_train[t]
        acc_train = test_accuracy(w, x_train, y_train)
        acc_valid = test_accuracy(w, x_valid, y_valid)
        print("iter %d, accuracy_train = %f, accuracy_valid = %f" % (i, acc_train, acc_valid))
    return w


def predict(w, x):
    res = np.matmul(w, np.transpose(x))
    vect_sign = np.vectorize(np.sign)
    return vect_sign(res)


def test_accuracy(w, x, y):
    pre_res = predict(w, x)
    sum_diff = 0
    for i in range(len(pre_res)):
        if pre_res[i] != y[i]:
            sum_diff += 1
    return 1 - sum_diff / len(y)


if __name__ == '__main__':
    sample_train, label_train = load_data("pa2_train.csv")
    sample_valid, label_valid = load_data("pa2_valid.csv")

    w = online_training(sample_train, label_train, sample_valid, label_valid)
    # print(w)
