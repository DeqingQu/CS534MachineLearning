import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import argparse
# import seaborn as sns
# import csv
import datetime


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


def gini_index(data):
    unique, count = np.unique(data, return_counts=True)
    counts = dict(zip(unique, count))
    uncertainty = 1
    for label in counts:
        prob_of_label = counts[label] / float(len(data))
        uncertainty -= prob_of_label**2
    return uncertainty


def info_gain(left, right, current_uncertainty):
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini_index(left) - (1 - p) * gini_index(right)


def split(data):
    best_gain = 0
    best_feature = -1
    best_threshold = 0
    current_uncertainty = gini_index(data[:, 0])
    n_features = data.shape[1]
    #   loop all features
    for i in range(1, n_features):
        data = data[data[:, i].argsort()]
        y_label = data[:, 0]
        feature_i = data[:, i]

        pre_label = 0
        #   loop all values in feature i
        for j, val in enumerate(feature_i):
            if pre_label == y_label[j]:
                continue
            pre_label = y_label[j]
            left = y_label[feature_i <= val]
            right = y_label[feature_i > val]
            if len(left) == 0 or len(right) == 0:
                continue
            gain = info_gain(left, right, current_uncertainty)
            if gain > best_gain:
                best_gain = gain
                best_feature = i
                best_threshold = val
    return best_gain, best_feature, best_threshold


class DecisionNode(object):
    def __init__(self, data, feature_idx, threshold):
        self.data = data
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.true_child = None
        self.false_child = None
        #   label of decision node
        unique, count = np.unique(data, return_counts=True)
        counts = dict(zip(unique, count))
        max_label = None
        max_count = 0
        for label in counts:
            if counts[label] > max_count:
                max_label = label
                max_count = counts[label]
        self.label = max_label


class DecisionLeaf(object):
    def __init__(self, data):
        unique, count = np.unique(data, return_counts=True)
        counts = dict(zip(unique, count))
        max_label = None
        max_count = 0
        for label in counts:
            if counts[label] > max_count:
                max_label = label
                max_count = counts[label]
        self.label = max_label


def partition(data, feature_idx, threshold):
    feature_i = data[:, feature_idx]
    true_rows = data[feature_i >= threshold]
    false_rows = data[feature_i < threshold]
    return true_rows, false_rows


def build_tree(data, height, max_depth=20):
    gain, f_idx, t = split(data)
    if gain == 0 or height >= max_depth:
        leaf = DecisionLeaf(data)
        return leaf
    true_data, false_data = partition(data, f_idx, t)
    node = DecisionNode(data, f_idx, t)
    if len(true_data) > 0:
        node.true_child = build_tree(true_data, height+1)
    if len(false_data) > 0:
        node.false_child = build_tree(false_data, height+1)
    return node


def classify(row, node, depth, max_depth=20):
    if node is None:
        return None
    if isinstance(node, DecisionLeaf):
        return node.label
    if depth >= max_depth:
        return node.label
    f_idx = node.feature_idx
    t = node.threshold
    if row[f_idx] >= t:
        return classify(row, node.true_child, depth+1, max_depth)
    else:
        return classify(row, node.false_child, depth+1, max_depth)


#   accuracy of decision tree
def validation(data, root, max_depth=20):
    error_count = 0
    for i in range(len(data)):
        label = classify(data[i], root, 0, max_depth)
        if label != data[i][0]:
            error_count += 1
    return 1 - float(error_count / len(data))


def decision_tree():
    data_train = load_data("pa3_train_reduced.csv")
    data_valid = load_data("pa3_valid_reduced.csv")
    now = datetime.datetime.now()
    dt_root = build_tree(data_train, 0)
    print("time to build a tree: ", datetime.datetime.now() - now)
    now = datetime.datetime.now()
    print("accuracy on the train set:")
    for depth in range(1, 21):
        print("depth %d, accuracy %f" % (depth, validation(data_train, dt_root, depth)))
    print("time to validate on the train set: ", datetime.datetime.now() - now)
    print("accuracy on the validation set:")
    for depth in range(1, 21):
        print("depth %d, accuracy %f" % (depth, validation(data_valid, dt_root, depth)))
    print("time to validate on the valid set:", datetime.datetime.now() - now)


if __name__ == '__main__':
    decision_tree()

    # print(data_train[:4, :10])
    # print(classify(data_train[0, :20], root))
    # print(classify(data_train[1, :20], root))
    # print(classify(data_train[2, :20], root))
    # print(classify(data_train[3, :20], root))
