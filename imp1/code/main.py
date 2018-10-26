import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv


def load_data(filename, has_label=True):
    """
    pre-process data
    :param filename: the path of the data file
    :param has_label: True if the data file has label column
    :return: matrix
    """
    tmp = np.loadtxt(filename, dtype=np.str, delimiter=",")
    #   remove the title row
    tmp = np.delete(tmp, 0, axis=0)

    #   remove the id column
    tmp = np.delete(tmp, 1, axis=1)

    #   split the data column to three different features: day, month, year
    tmp = np.insert(tmp, 2, values='year', axis=1)
    tmp = np.insert(tmp, 2, values='day', axis=1)
    tmp = np.insert(tmp, 2, values='month', axis=1)
    for i in range(len(tmp)):
        date = tmp[i][1].split('/')
        tmp[i][2], tmp[i][3], tmp[i][4] = date[0], date[1], date[2]

        #   process the year of renovated
        #   if the year of renovated is 0, set it as the year of built
        if tmp[i][17] == '0':
            tmp[i][17] = tmp[i][16]

        #   process the bedrooms, if more than 10, set it to ten
        if float(tmp[i][5]) > 10:
            tmp[i][5] = 10

        #   process the zip code
        #   if the zip code starts with 981, set it as category 1
        #   else if it starts with 980, set it as category 0
        if tmp[i][18][:3] == '981':
            tmp[i][18] = 1
        elif tmp[i][18][:3] == '980':
            tmp[i][18] = 0
        else:
            tmp[i][18] = -1

    #   remove the original data column
    tmp = np.delete(tmp, 1, axis=1)

    #   transfer the dtype of the matrix from string to float
    tmp = tmp.astype(float)

    # # boost the sqrt living, condition, grade, sqrt living 15
    # tmp = np.insert(tmp, 22, values=1.0, axis=1)
    # tmp = np.insert(tmp, 22, values=1.0, axis=1)
    # tmp = np.insert(tmp, 22, values=1.0, axis=1)
    # tmp = np.insert(tmp, 22, values=1.0, axis=1)
    # for i in range(len(tmp)):
    #     tmp[i][22] = tmp[i][6] ** 2     # sqrt living
    #     tmp[i][23] = tmp[i][11] ** 2    # condition
    #     tmp[i][24] = tmp[i][12] ** 2    # grade
    #     tmp[i][25] = tmp[i][20] ** 2    # sqrt living 15

    #   generate basement / above ratio
    #   generate living15 / lot15 ratio
    # tmp = np.insert(tmp, 22, values=1.0, axis=1)
    # tmp = np.insert(tmp, 22, values=1.0, axis=1)

    if has_label:
        return tmp[:, :-1], tmp[:, -1]
    else:
        return tmp


def report_statistics(data):
    """
    print the statistics of the data set after pre-process
    :param data: matrix
    :return:
    """
    # calculate the mean, the standard deviation, the range for numerical features
    print("mean : " + str(np.mean(data, axis=0)))
    print("std : " + str(np.std(data, axis=0)))
    print("range : " + str(np.ptp(data, axis=0)))
    # print(np.min(data, axis=0))
    # print(np.max(data, axis=0))
    # calculate the percentages of examples for category features
    print("waterfront %s" % calculate_percentage(data[:, 9]))     # waterfront
    print("condition %s" % calculate_percentage(data[:, 11]))    # condition
    print("grade %s" % calculate_percentage(data[:, 12]))    # grade
    print("zip code %s" % calculate_percentage(data[:, 17]))    # zip code


def normalize(v):
    """
    normalize a vector
    :param v:
    :return:
    """
    if np.ptp(v) == 0:
        if v[0] == 0:
            return v
        return v / v[0]
    return (v - v.min()) / (np.ptp(v))
    # norm = np.linalg.norm(v)
    # if norm == 0:
    #    return v
    # return v / norm


def normalize_matrix(data):
    """
    normalize a matrix according to each column
    :param data: matrix
    :return:
    """
    if data is None or len(data) == 0:
        return data
    for i in range(len(data[0])):
        data[:, i] = normalize(data[:, i])
    return data


def calculate_percentage(data):
    """
    calculate the percentage of examples in each category
    :param data: vector (column in matrix)
    :return: dictionary with category as key and percentage as value
        {'category1': percentage1, 'category2': percentage2, ...}
    """
    dic = {}
    for val in data:
        if val in dic:
            dic[val] += 1
        else:
            dic[val] = 0
    total = 0
    for key in dic.keys():
        total += dic[key]
    for key in dic.keys():
        dic[key] = float(dic[key] / total)
    return dic


def gradient_descent(x, y, lr, lamda, iterations, batch_size):
    """
    using gradient descent algorithm to optimize the SSE

    det w = sum(xi * (yi' - yi)) + lamda * w

    :param x:   matrix of training samples
    :param y:   vector of training labels
    :param lr:  learning rate of training
    :param lamda:   lamda for regularization
    :param iterations:  the maximum number of iterations
    :param batch_size:  the batch size for mini-batch
    :return:    weights
    """
    w = np.random.uniform(-0.2, 0.2, len(x[0]))

    batch_count = len(x) // batch_size

    for it in range(iterations+1):
        for batch_i in range(batch_count):

            p_y = np.matmul(x[batch_i * batch_size: (batch_i + 1) * batch_size], w)
            det_w = np.matmul(np.transpose(x[batch_i * batch_size: (batch_i + 1) * batch_size]), p_y - y)
            #   add the regularization item
            w_for_reg = np.array(w)
            w_for_reg[0] = 0
            det_w += lamda * w_for_reg
            norm = np.linalg.norm(det_w)

            #   threshold for end iteration
            if norm <= 0.5:
                print("iteration ends : " + str(it))
                return w

            #   print debug information
            if (it % 1000 == 0) and batch_i == batch_count - 1:
                sse = calculate_loss(x, y, w)
                print("it = %d, SSE = %f" % (it, sse))
                # print("det w = " + str(det_w))
                print("norm = %f" % norm)

            w -= lr * det_w
    return w


def calculate_loss(x, y, w):
    """
    calculate the SSE

    loss = sum((yi - w * xi)**2) + lamda * w**2

    :param x:   matrix of training samples
    :param y:   vector of training labels
    :param w:   weights
    :param lamda:   lamda for regularization
    :return:    SSE value
    """
    loss = 0
    for i in range(len(x)):
        yi = np.dot(w, x[i])
        loss += (y[i] - yi)**2
    return loss  # / len(x)


def predict(x, w):
    """
    predict the labels of training sample, according to the weights
    :param x:   matrix of training samples
    :param w:   weights
    :return:    vector of training labels
    """
    y = []
    for i in range(len(x)):
        y.append(np.dot(w, x[i]))
    return y


if __name__ == '__main__':
    learning_rate = 0.00001
    lamda = 1
    max_iterations = 1000000
    batch_size = 10000
    train_data, train_label = load_data('PA1_train.csv')

    #   plot features
    features = [6, 7, 13, 14, 18, 19, 20, 21]
    for i in features:
        plt.figure(figsize=(15, 6))
        plt.subplot(1, 2, 1)
        fig = sns.boxplot(y=train_data[:, i])
        fig.set_title('')
        fig.set_ylabel(i)

        plt.subplot(1, 2, 2)
        df = pd.DataFrame(train_data[:, i])
        fig = sns.distplot(df.dropna())
        fig.set_ylabel('Number of houses')
        fig.set_xlabel(i)

        plt.show()

    # report_statistics(train_data)
    train_data = normalize_matrix(train_data)

    weights = gradient_descent(train_data, train_label, learning_rate, lamda, max_iterations, batch_size)
    print("weight = " + str(weights))

    validate_data, validate_label = load_data('PA1_dev.csv')
    validate_data = normalize_matrix(validate_data)
    print("SSE on validation dataset: %f" % (calculate_loss(validate_data, validate_label, weights)))

    test_data = load_data('PA1_test.csv', has_label=False)
    test_data = normalize_matrix(test_data)
    test_labels = predict(test_data, weights)

    #   output result to csv
    with open('predict_test.csv', mode='w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(test_labels)):
            writer.writerow([test_labels[i]])
