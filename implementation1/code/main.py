import numpy as np


def load_data(filename, has_label=True):
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
        date = tmp[i][1].split('/')
        tmp[i][2], tmp[i][3], tmp[i][4] = date[0], date[1], date[2]
        #   process the year of renovated
        if tmp[i][17] == '0':
            tmp[i][17] = tmp[i][16]
        #   process the zip code
        if tmp[i][18][:3] == '981':
            tmp[i][18] = 1
        elif tmp[i][18][:3] == '980':
            tmp[i][18] = 0
        else:
            tmp[i][18] = -1
    tmp = np.delete(tmp, 1, axis=1)

    #   string to int or float
    tmp = tmp.astype(float)

    if has_label:
        return tmp[:, :-1], tmp[:, -1]
    else:
        return tmp


def report_statistics(data):
    # calculate the mean, the standard deviation, the range for numerical features
    print(np.mean(data, axis=0))
    print(np.std(data, axis=0))
    print(np.ptp(data, axis=0))
    # print(np.min(data, axis=0))
    # print(np.max(data, axis=0))
    # calculate the percentages of examples for category features
    print("waterfront %s" % calculate_percentage(data[:, 9]))     # waterfront
    print("condition %s" % calculate_percentage(data[:, 11]))    # condition
    print("grade %s" % calculate_percentage(data[:, 12]))    # grade
    print("zip code %s" % calculate_percentage(data[:, 17]))    # zip code


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm


def normalize_all_columns(data):
    if data is None or len(data) == 0:
        return data
    for i in range(len(data[0])):
        data[:, i] = normalize(data[:, i])
    return data


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
    print(train_data[0])
    print(train_label)

    report_statistics(train_data)

    train_data = normalize_all_columns(train_data)
    print(train_data[0])