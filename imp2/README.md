Language: Python 3.5

Packages: Numpy, csv, argparse

How to run Part 1: Online Perceptron  
python3 main.py --run op

Output example:  
iter 1, accuracy_train = 0.947422, accuracy_valid = 0.933702  
iter 2, accuracy_train = 0.957651, accuracy_valid = 0.945365  
iter 3, accuracy_train = 0.949877, accuracy_valid = 0.934929  
iter 4, accuracy_train = 0.964812, accuracy_valid = 0.948435  
...

How to run Part 2: Average Perceptron  
python3 main.py --run ap

Output example:  
iter 1, accuracy_train = 0.956424, accuracy_valid = 0.944751  
iter 2, accuracy_train = 0.961743, accuracy_valid = 0.949662  
iter 3, accuracy_train = 0.963175, accuracy_valid = 0.951504  
iter 4, accuracy_train = 0.966244, accuracy_valid = 0.950890  
...

How to run Part 3: Kernel Perceptron  
python3 main.py --run kp

Output example:  
P = 3  
iter 0, accuracy_train = 0.978928, accuracy_valid = 0.967465  
iter 1, accuracy_train = 0.985679, accuracy_valid = 0.969306  
iter 2, accuracy_train = 0.988134, accuracy_valid = 0.974217  
iter 3, accuracy_train = 0.986702, accuracy_valid = 0.971148  
...