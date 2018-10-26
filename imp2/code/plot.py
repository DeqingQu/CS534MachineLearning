import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    P = [1, 2, 3, 7, 15]
    accuracy = [0.948435, 0.983425, 0.984653, 0.977287, 0.965623]
    plt.figure()
    plt.plot(P, accuracy, "r-", linewidth=1)
    plt.xlabel("Degree")
    plt.ylabel("Best Validation Accuracy")
    plt.title("Best Validation Accuracy vs. Degree")
    plt.show()