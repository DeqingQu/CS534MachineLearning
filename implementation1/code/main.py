import numpy as np


def load_data(filename):
    tmp = np.loadtxt('data/' + filename, dtype=np.str, delimiter=",")
    #   remove the title row
    tmp = np.delete(tmp, 0, axis=0)

    #   remove the id column
    tmp = np.delete(tmp, 1, axis=1)

    #   split the data column to three different features: day, month, year
    tmp = np.insert(tmp, 2, values='year', axis=1)
    tmp = np.insert(tmp, 2, values='day', axis=1)
    tmp = np.insert(tmp, 2, values='month', axis=1)
    for i in range(len(tmp)):
        tmp[i][2], tmp[i][3], tmp[i][4] = split_date(tmp[i][1])
    tmp = np.delete(tmp, 1, axis=1)

    #   string to int or float
    tmp = tmp.astype(float)

    return tmp[:, :-1], tmp[:, -1]


def split_date(date):
    temp = date.split('/')
    return temp[0], temp[1], temp[2]


def report_statistics(data):
    # calculate the mean, the standard deviation, the range for numerical features
    print(np.mean(data, axis=0))
    print(np.std(data, axis=0))
    print(np.ptp(data, axis=0))
    # print(np.min(data, axis=0))
    # print(np.max(data, axis=0))
    # calculate the percentages of examples for category features
    print(calculate_percentage(data[:, 9]))
    print(calculate_percentage(data[:, 11]))
    print(calculate_percentage(data[:, 12]))


def calculate_percentage(data):
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


if __name__ == '__main__':
    train_data, train_label = load_data('PA1_train.csv')
    print(train_data)
    print(train_label)

    report_statistics(train_data)
    # report_statistics(train_label)