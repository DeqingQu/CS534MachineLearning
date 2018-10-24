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


def online_perceptron(x_train, y_train, x_valid, y_valid, iters=15):
    w = np.zeros(len(x_train[0]))
    acc_train = []
    acc_valid = []
    for i in range(iters):
        for t in range(len(x_train)):
            u = np.sign(x_train[t].dot(w))
            if u * y_train[t] <= 0:
                w += y_train[t] * x_train[t]
        acc_train.append(test_accuracy(w, x_train, y_train))
        acc_valid.append(test_accuracy(w, x_valid, y_valid))
        print("iter %d, accuracy_train = %f, accuracy_valid = %f" % (i, acc_train[i], acc_valid[i]))
    return w, acc_train, acc_valid


def average_perceptron(x_train, y_train, x_valid, y_valid, iters=15):
    w = np.zeros(len(x_train[0]))
    w_a = np.zeros(len(x_train[0]))
    acc_train, acc_valid = [], []
    c, s = 0, 0
    for i in range(iters):
        for t in range(len(x_train)):
            u = np.sign(x_train[t].dot(w))
            if u * y_train[t] <= 0:
                if s + c > 0:
                    w_a = (s * w_a + w * c) / (s + c)
                s = s + c
                w += y_train[t] * x_train[t]
                c = 0
            else:
                c += 1
        if c > 0:
            w_a = (s * w_a + w * c) / (s + c)
        acc_train.append(test_accuracy(w_a, x_train, y_train))
        acc_valid.append(test_accuracy(w_a, x_valid, y_valid))
        print("iter %d, accuracy_train = %f, accuracy_valid = %f" % (i, acc_train[i], acc_valid[i]))
    return w, acc_train, acc_valid


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


def plot_accuracy(acc_train, acc_valid):
    iters = range(1, len(acc_train)+1)
    plt.figure()
    plt.plot(iters, acc_train, "b-", linewidth=1)
    plt.plot(iters, acc_valid, "r-", linewidth=1)
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Iterations")
    plt.show()


if __name__ == '__main__':
    sample_train, label_train = load_data("pa2_train.csv")
    sample_valid, label_valid = load_data("pa2_valid.csv")

    #   Online Perceptron
    # w_op, acc_t_op, acc_v_op = online_perceptron(sample_train, label_train, sample_valid, label_valid, iters=15)
    # plot_accuracy(acc_t_op, acc_v_op)

    #   Average Perceptron
    w_ap, acc_t_ap, acc_v_ap = average_perceptron(sample_train, label_train, sample_valid, label_valid, iters=50)
    plot_accuracy(acc_t_ap, acc_v_ap)

