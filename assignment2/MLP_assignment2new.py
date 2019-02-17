

from __future__ import division
from __future__ import print_function
import sys
import cPickle
import numpy as np

import random
import time
from math import exp, log, e
from copy import deepcopy
import matplotlib
import matplotlib.pyplot as plt

# This is a class for a LinearTransform layer which takes an input
# weight matrix W and computes W x as the forward step
class LinearTransform(object):
    def __init__(self, input_dims, hidden_units):
    # Define_init function
        self.d = input_dims
        self.m = hidden_units
        self.W = np.random.randn(self.d, self.m)/10 
        self.b = np.random.randn(self.m,1)/10   
        self.batch = 0
        self.layer_input = []
        self.layer_output = 0
        self.back_w, self.back_b, self.back_x = 0,0,0
        self.dw, self.db = 0,0
    def forward(self, x, size_batch):              
    # DEFINE forward function
        self.batch = size_batch
        self.dw = np.zeros((self.d, self.m))
        self.db = np.zeros((self.m,1))
        self.layer_input = x.reshape(self.d,self.batch)
        self.layer_output = np.dot(self.W.T,self.layer_input) + self.b    
    def backward(
        self,
        grad_output,
        learning_rate=0.0,
        momentum=0.0,
        l2_penalty=0.0,
    ):
    # DEFINE backward function
        self.back_w = np.zeros((self.d, self.m))
        for j in xrange(self.batch):
            self.back_w += np.dot(self.layer_input[:,j].reshape(self.d,1),grad_output[:,j].reshape(1,self.m)) 
        self.back_w = self.back_w / self.batch

        self.back_b = np.mean(grad_output,axis = 1).reshape(-1,1)       
        self.back_x = np.dot(self.W, grad_output)           
        self.dw = momentum * self.dw - learning_rate * self.back_w    
        self.db = momentum * self.db - learning_rate * self.back_b    
        self.W += self.dw
        self.b += self.db

# ADD other operations in LinearTransform if needed
# This is a class for a ReLU layer max(x,0)
class ReLU(object):
    def __init__(self, x=0):
        self.layer_input = x
        self.layer_output = []
        self.back = 0

    def forward(self, x):               
    # DEFINE forward function
        self.layer_input = x                
        self.layer_output = deepcopy(x)      
        self.layer_output[self.layer_output < 0] = 0

    def backward(
    # DEFINE backward function
        self,
        grad_output,
        learning_rate=0.0,
        momentum=0.0,
        l2_penalty=0.0,
    ):
        y = deepcopy(self.layer_input)     
        y[y > 0] = 1
        y[y == 0] = 0.5
        y[y < 0] = 0
        self.back = grad_output * y    

# ADD other operations in ReLU if needed

# This is a class for a sigmoid layer followed by a cross entropy layer, the reason
# this is put into a single layer is because it has a simple gradient form
class SigmoidCrossEntropy(object):
    def __init__(self, x=0):
        self.layer_input = x
        self.layer_output = 0
        self.label = 0
        self.loss = 0
        self.back = 0
    def forward(self, x,y):
        # DEFINE forward function
        x = x.reshape(-1)       
        y = y.reshape(-1)      

        self.layer_input = x
        temp_exp = e ** (-np.absolute(x))   
        self.layer_output = np.where(x >= 0,1/(1+temp_exp),temp_exp/(1+temp_exp))   
        self.loss = np.where(x<0,0,x) - y * x + np.log(1+temp_exp)  
        self.label = y

    def backward(
        self,
        #grad_output,
        learning_rate=0.0,
        momentum=0.0,
        l2_penalty=0.0
    ):
        # DEFINE backward function
        self.back = (self.layer_output-self.label).reshape(1,-1)        
# ADD other operations and data entries in SigmoidCrossEntropy if needed


class MLP(object):
    def __init__(self,input_dims, hidden_units):
        #Insert code for initializing the network
        self.l1 = LinearTransform(input_dims, hidden_units)
        self.l2 = ReLU()
        self.l3 = LinearTransform(hidden_units, 1)
        self.l4 = SigmoidCrossEntropy()
    def forward(self,x,label,size_batch):
        self.l1.forward(x, size_batch)
        self.l2.forward(self.l1.layer_output)
        self.l3.forward(self.l2.layer_output, size_batch)
        self.l4.forward(self.l3.layer_output,label)
    def backward(self,learning_rate=0.0, momentum=0.0):
        self.l4.backward()
        self.l3.backward(self.l4.back,learning_rate, momentum,l2_penalty=0.0)
        self.l2.backward(self.l3.back_x)
        self.l1.backward(self.l2.back,learning_rate, momentum,l2_penalty=0.0)
    def evaluate(self):
        predic = deepcopy(self.l4.layer_output)
        predic[predic >= 0.5] = 1
        predic[predic < 0.5]  = 0
        return np.sum(np.absolute(predic-self.l4.label))


def normalize(X):    
    Xmean  = np.mean(X,axis=0).reshape(1,-1)     
    Xstd   = np.std(X,axis=0).reshape(1,-1)      
    normalized_X  = (X - Xmean) / Xstd
    return normalized_X


def Main(train_x,train_y,
            test_x,test_y,
            num_epochs,
            size_batch,
            hidden_units,
            lr,mu,
            ):
    print("size_batch: " + str(size_batch))
    print("hidden_units: " + str(hidden_units))
    print("lr: " + str(lr))

    num_examples, input_dims = train_x.shape
    num_test = test_x.shape[0]
    #size_batch = int(num_examples / num_batches)
    n = MLP(input_dims,hidden_units)

    acc1, acc2, loss1, loss2 = [],[],[],[]
    for epoch in xrange(num_epochs):
    # INSERT YOUR CODE FOR EACH EPOCH HERE
        train_loss = 0.0
        start_line = 0
        for b in xrange(int(num_examples/size_batch)):
            # INSERT YOUR CODE FOR EACH MINI_BATCH HERE
            # MAKE SURE TO UPDATE total_loss
            total_loss = 0.0
            batch_x = train_x[start_line:(start_line+size_batch),:]
            batch_y = train_y[start_line:(start_line+size_batch)]
            start_line += size_batch
            n.forward(batch_x.T,batch_y,size_batch)
            n.backward(lr,mu)
            total_loss = np.sum(n.l4.loss)
            train_loss += total_loss

            print(
                '\r[Epoch {}, mb {}]    Avg.Loss = {:.3f}'.format(
                    epoch + 1,
                    b + 1,
                    total_loss/size_batch,
                ),
                end='',
            )
            sys.stdout.flush()
        # MAKE SURE TO COMPUTE train_loss, train_accuracy, test_loss, test_accuracy

        n.forward(train_x.T,train_y,num_examples)  
        train_loss = np.sum(n.l4.loss)/num_examples
        error = n.evaluate()
        train_accuracy = 1 - error*1.0/num_examples
    

        n.forward(test_x.T,test_y,num_test) 
        test_loss = np.sum(n.l4.loss)/num_test
        error = n.evaluate()
        test_accuracy = 1 - error*1.0/num_test
    

        print()
        print('    Train Loss: {:.3f}    Train Acc: {:.2f}%     Error Rate: {:.2f}% ' .format(
            train_loss,
            100. * train_accuracy,
            100*(1 - train_accuracy)

        ))
        print('    Test Loss:  {:.3f}    Test Acc:  {:.2f}%     Error Rate: {:.2f}% ' .format( 
            test_loss,
            100. * test_accuracy,
            100*(1 - test_accuracy)
        ))
        acc1.append(train_accuracy)     
        acc2.append(test_accuracy)     
        loss1.append(train_loss)        
        loss2.append(test_loss)        
    return acc1,acc2,loss1,loss2


if __name__ == '__main__':

    t0 = t000 = float(time.clock())
    data = cPickle.load(open('cifar_2class_py2.p', 'rb'))
    t1 = float(time.clock())
    print ('Loading time is %.4f s. \n' % (t1-t0))

    train_x = normalize(data['train_data'])
    test_x = normalize(data['test_data'])
    train_y = data['train_labels']
    test_y = data['test_labels']    

    num_epochs = 40

    size_batch = 50       #fixed           
    batch_list = [50,200,1000]    

    hidden_units = 100       #fixed          
    hiddenU_list = [10,100,200]        

    lr = 0.005         #fixed              
    lr_list = [0.05,0.01,0.005]       
    momentum = 0.8       

    #print("size_batch: " + str(size_batch))
    #print("hidden_units: " + str(hidden_units))
    #print("lr: " + str(lr))

    '''tune learning rate '''
    t1 = t00 = float(time.clock())
    acc_train_mat, acc_test_mat, loss_train_mat, loss_test_mat = [],[],[],[]
    for l_r in lr_list:
        t0 = t1
        acc_train,acc_test, loss_train, loss_test = Main(train_x,train_y,
                test_x,test_y,
                num_epochs,
                size_batch,
                hidden_units,
                l_r,momentum
                )
        acc_train_mat.append(acc_train)
        acc_test_mat.append(acc_test)
        loss_train_mat.append(loss_train)
        loss_test_mat.append(loss_test)

        t1 = float(time.clock())
        para = 'hidden units: %d, size_batch: %d, learning rate: %.4f, momentum: %.1f' %(hidden_units,size_batch,l_r,momentum)
        print (para)
        print ('Running time is %.4f s. \n' % (t1-t0))

    t1 = float(time.clock())
    print ('Time for tuning learning rate is %.4f s. \n' % (t1-t00))
    #lr_list = [0.05,0.01,0.005] 
    plt.figure(1)
    plt.plot(range(1,num_epochs+1),acc_test_mat[0],color="green", linewidth=1.5, linestyle="-",label ="lr = 0.05")
    plt.plot(range(1,num_epochs+1),acc_test_mat[1],color="red", linewidth=1.5, linestyle="-",label ="lr = 0.01")
    plt.plot(range(1,num_epochs+1),acc_test_mat[2],color="blue", linewidth=1.5, linestyle="-",label ="lr = 0.005")
    plt.xlim(1, num_epochs)
    plt.ylim(0.5, 1)
    plt.xlabel('epoch')
    plt.ylabel('Test Accuracy')
    plt.legend(loc='upper left')
    plt.show()


    ''' tune hidden units '''
    t1 = t00 = float(time.clock())
    acc_train_mat, acc_test_mat, loss_train_mat, loss_test_mat = [],[],[],[]
    for hidden_u in hiddenU_list:
        t0 = t1
        acc_train,acc_test, loss_train, loss_test = Main(train_x,train_y,
                test_x,test_y,
                num_epochs,
                size_batch,
                hidden_u,
                lr,momentum
                )
        acc_train_mat.append(acc_train)
        acc_test_mat.append(acc_test)
        loss_train_mat.append(loss_train)
        loss_test_mat.append(loss_test)

        t1 = float(time.clock())
        para = 'hidden units: %d, size_batch: %d, learning rate: %.4f, momentum: %.1f' %(hidden_u,size_batch,lr,momentum)
        print (para)
        print ('Running time is %.4f s. \n' % (t1-t0))
    t1 = float(time.clock())
    print ('Time for tuning hidden units is %.4f s. \n' % (t1-t00))

    #hiddenU_list = [50,100,200] 
    plt.figure(2)
    plt.plot(range(1,num_epochs+1),acc_test_mat[0],color="green", linewidth=1.5, linestyle="-",label ="hidden_units = 10")
    plt.plot(range(1,num_epochs+1),acc_test_mat[1],color="red", linewidth=1.5, linestyle="-",label ="hidden_units = 100")
    plt.plot(range(1,num_epochs+1),acc_test_mat[2],color="blue", linewidth=1.5, linestyle="-",label ="hidden_units = 200")
    plt.xlim(1, num_epochs)
    plt.ylim(0.5, 1)
    plt.xlabel('epoch')
    plt.ylabel('Test Accuracy')
    plt.legend(loc='upper left')
    plt.show()

    ''' tune batches '''
    t1 = t00 = float(time.clock())
    acc_train_mat, acc_test_mat, loss_train_mat, loss_test_mat = [],[],[],[]
    for batch in batch_list:
        t0 = t1
        acc_train,acc_test, loss_train, loss_test = Main(train_x,train_y,
                test_x,test_y,
                num_epochs,
                size_batch,
                hidden_units,
                lr,momentum
                )
        acc_train_mat.append(acc_train)
        acc_test_mat.append(acc_test)
        loss_train_mat.append(loss_train)
        loss_test_mat.append(loss_test)

        t1 = float(time.clock())
        para = 'hidden units: %d, size_batch: %d, learning rate: %.4f, momentum: %.1f' %(hidden_units,size_batch,lr,momentum)
        print (para)
        print ('Running time is %.4f s. \n' % (t1-t0))
    t1 = float(time.clock())
    print ('Time for tuning batches is %.4f s. \n' % (t1-t00))
    

    #batch_list = [50,200,1000]
    plt.figure(3)
    plt.plot(range(1,num_epochs+1),acc_test_mat[0],color="green", linewidth=1.5, linestyle="-",label ="batch size = 50")
    plt.plot(range(1,num_epochs+1),acc_test_mat[1],color="red", linewidth=1.5, linestyle="-",label ="batch size = 200")
    plt.plot(range(1,num_epochs+1),acc_test_mat[2],color="blue", linewidth=1.5, linestyle="-",label ="batch size = 1000")

    plt.xlim(1, num_epochs)
    plt.ylim(0.5, 1)
    plt.xlabel('epoch')
    plt.ylabel('Test Accuracy')
    plt.legend(loc='upper left')
    plt.show()
    