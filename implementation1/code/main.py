import numpy as np


def load_data(filename):
    tmp = np.loadtxt('data/' + filename, dtype=np.str, delimiter=",")
    #   remove the id column
    tmp = np.delete(tmp, 1, axis=1)
    #   split the data column to three different features: day, month, year
    tmp = np.insert(tmp, 2, values='year', axis=1)
    tmp = np.insert(tmp, 2, values='day', axis=1)
    tmp = np.insert(tmp, 2, values='month', axis=1)
    for i in range(1, len(tmp)):
        tmp[i][2], tmp[i][3], tmp[i][4] = split_date(tmp[i][1])
    tmp = np.delete(tmp, 1, axis=1)
    return tmp


def split_date(date):
    temp = date.split('/')
    return temp[0], temp[1], temp[2]


if __name__ == '__main__':
    train_data = load_data('PA1_train.csv')
    print(train_data)