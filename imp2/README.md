Language: Python 3.5

Packages: Numpy, csv, argparse

How to run：

**Part 1: Online Perceptron**  
python3 main.py --run op -i 15

Output example:  
iteration number	accuracy on the training set	accuracy on the validation set  
1	0.947422	0.933702  
2	0.957651	0.945365  
3	0.949877	0.934929  
...

**Part 2: Average Perceptron**  
python3 main.py --run ap -i 15

Output example:  
iteration number	accuracy on the training set	accuracy on the validation set  
1	0.956424	0.944751  
2	0.961743	0.949662  
3	0.963175	0.951504  
...

**How to run Part 3: Kernel Perceptron**  
python3 main.py --run kp -i 15 -p 3

Output example:  
iteration number	accuracy on the training set	accuracy on the validation set  
1	0.980769	0.969920  
2	0.992840	0.980970  
3	0.998363	0.981584  
...