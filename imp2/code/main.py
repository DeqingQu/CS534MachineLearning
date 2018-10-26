import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
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


def online_perceptron(x_train, y_train, x_valid, y_valid, iters=15):
    w = np.zeros(len(x_train[0]))
    acc_train, acc_valid = [], []
    for i in range(iters):
        for t in range(len(x_train)):
            u = np.sign(x_train[t].dot(w))
            if u * y_train[t] <= 0:
                w += y_train[t] * x_train[t]
        acc_train.append(test_accuracy(w, x_train, y_train))
        acc_valid.append(test_accuracy(w, x_valid, y_valid))
        print("iter %d, accuracy_train = %f, accuracy_valid = %f" % (i+1, acc_train[i], acc_valid[i]))
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
        print("iter %d, accuracy_train = %f, accuracy_valid = %f" % (i+1, acc_train[i], acc_valid[i]))
    return w, acc_train, acc_valid


def kernel_function(x, y, p):
    return (1 + np.dot(x, y)) ** p


def kernel_function_3(x, y):
    return (1 + np.dot(x, y)) ** 3


def kernel_perceptron(x_train, y_train, x_valid, y_valid, p, iters=15):
    acc_train, acc_valid = [], []
    N = len(x_train)
    alpha = np.zeros(N)

    #   Gram Matrix
    K = np.zeros((N, N))
    for i in range(N):
        if i % 500 == 0:
            print("i = %d" % i)
        for j in range(N):
            K[i, j] = kernel_function(x_train[i], x_train[j], p)

    for it in range(iters):
        for i in range(N):
            u = 0
            for j in range(N):
                u += K[j, i] * alpha[j] * y_train[j]
            u = np.sign(u)
            if y_train[i] * u <= 0:
                alpha[i] += 1

        pred = predict_kernel(x_train, alpha, x_train, y_train, p)
        acc_train.append(test_accuracy_kernel(pred, y_train))
        pred = predict_kernel(x_valid, alpha, x_train, y_train, p)
        acc_valid.append(test_accuracy_kernel(pred, y_valid))
        print("P = %d\niter %d, accuracy_train = %f, accuracy_valid = %f" % (p, it+1, acc_train[it], acc_valid[it]))
    return alpha, acc_train, acc_valid


def predict_kernel(x_test, alpha, x_train, y_train, p):
    res = []
    for t in range(len(x_test)):
        u = 0
        for j in range(len(x_train)):
            u += kernel_function(x_train[j], x_test[t], p) * alpha[j] * y_train[j]
        u = np.sign(u)
        res.append(u)
    return res


def test_accuracy_kernel(pre, y):
    sum_diff = 0
    for i in range(len(pre)):
        if pre[i] != y[i]:
            sum_diff += 1
    return 1 - sum_diff / len(y)


def plot_accuracy(acc_train, acc_valid, title):
    iters = range(1, len(acc_train)+1)
    plt.figure()
    plt.plot(iters, acc_train, "b-", linewidth=1, label="training accuracy")
    plt.plot(iters, acc_valid, "r-", linewidth=1, label="validation accuracy")
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Iterations (" + title + ")")
    plt.show()


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Perceptron Implementation')
    #
    # parser.add_argument('--run', help="run method: op (Online Perceptron), ap (Average Perceptron), kp (Kernel Perceptron)", default='')
    # parser.add_argument("-i", "--iterations", help="iteration number (default: 15)", default='15')
    # parser.add_argument("-p", "--p", help="p for the kernel function (default: 3)", default='3')
    # args = parser.parse_args()
    #
    # if args.run == '':
    #     print('usage: main.py [--runfunc RUNFUNC: op, ap, kp] [-p p]')
    #     print('main.py: error: invalid parameter')
    #     exit(0)
    #
    sample_train, label_train = load_data("pa2_train.csv")
    sample_valid, label_valid = load_data("pa2_valid.csv")
    #
    # if args.run == 'op':
    #     w_op, acc_t_op, acc_v_op = online_perceptron(sample_train, label_train, sample_valid, label_valid, iters=int(args.iterations))
    #     plot_accuracy(acc_t_op, acc_v_op, "Online Perceptron")
    # elif args.run == 'ap':
    #     w_ap, acc_t_ap, acc_v_ap = average_perceptron(sample_train, label_train, sample_valid, label_valid, iters=int(args.iterations))
    #     plot_accuracy(acc_t_ap, acc_v_ap, "Average Perceptron")
    # elif args.run == 'kp':
    #     a_kp, acc_t_kp, acc_v_kp = kernel_perceptron(sample_train, label_train, sample_valid, label_valid, p=int(args.p), iters=int(args.iterations))
    #     plot_accuracy(acc_t_kp, acc_v_kp, "Kernel Perceptron")

    # #   Online Perceptron
    # w_op, acc_t_op, acc_v_op = online_perceptron(sample_train, label_train, sample_valid, label_valid, iters=15)
    # plot_accuracy(acc_t_op, acc_v_op, "Online Perceptron")
    #
    # #   Average Perceptron
    # w_ap, acc_t_ap, acc_v_ap = average_perceptron(sample_train, label_train, sample_valid, label_valid, iters=15)
    # plot_accuracy(acc_t_ap, acc_v_ap, "Average Perceptron")
    #
    #   Kernel Perceptron
    a_kp, acc_t_kp, acc_v_kp = kernel_perceptron(sample_train[:500], label_train[:500], sample_valid, label_valid, p=3, iters=15)
    plot_accuracy(acc_t_kp, acc_v_kp, "Kernel Perceptron")

