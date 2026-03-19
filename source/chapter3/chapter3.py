import numpy as np
from matplotlib import pyplot as plt
plt.ion()

def step(x):
    y=x>0
    return y.astype(int)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    # rectified linear unit
    return np.maximum(0,x)

def identity(x):
    return x

def step_draw():
    x=np.arange(-3,3,0.1)
    y=step(x)
    plt.xlim(-1.1,1.1)
    plt.ylim(-0.1,1.1)
    plt.plot(x,y)
    plt.show()

def sigmoid_draw():
    x=np.arange(-3,3,0.1)
    y=sigmoid(x)
    plt.xlim(-1.1,1.1)
    plt.ylim(-0.1,1.1)
    plt.plot(x,y)
    plt.show()

def relu_draw():
    x=np.arange(-3,3,0.1)
    y=relu(x)
    plt.xlim(-1.1,1.1)
    plt.ylim(-0.1,1.1)
    plt.plot(x,y)
    plt.show()

def softmax(a):
    return np.exp(a-np.max(a))/np.sum(np.exp(a-np.max(a)))

def all_compare_draw():
    x=np.arange(-3,3,0.01)

    y1=step(x)
    y2=sigmoid(x)
    y3=relu(x)

    plt.ylim(-0.1,1.1)

    plt.plot(x,y1,linestyle='--',label='step')
    plt.plot(x,y2,label='sigmoid')
    plt.plot(x,y3,label='relu',linestyle='-.')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

def init_network():
    network={}
    network['W1']=np.array([[1,3,5],[2,4,6]])
    network['B1']=np.array([0.5,0.4,0.3])
    network['W2']=np.array([[1,3],[2,4],[5,7]])
    network['B2']=np.array([0.2,0.6])
    network['W3']=np.array([[4,5],[2,1]])
    network['B3']=np.array([0.2,0.6])
    return network

def process_forward(x):
    network=init_network()
    W1,B1,W2,B2,W3,B3=network['W1'],network['B1'],network['W2'],network['B2'],network['W3'],network['B3']

    return softmax(np.dot(sigmoid(np.dot(sigmoid(np.dot(x,W1)+B1),W2)+B2),W3)+B3)