import numpy as np

def step(x):
    return np.array(x > 0, dtype=np.int)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    # rectified linear unit
    grad = np.zeros(x)
    grad[x>=0] = 1
    return grad

def identity(x):
    return x

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

def mean_squared_error(y,t):
    return np.mean((y-t)**2)

def sum_squared_error(y,t):
    return np.sum((y-t)**2)

def cross_entropy_error(y,t,batch_size):
    if y.ndim==1:
        y.reshape(1,y.size)
        t.reshape(1,t.size)

    delta=1e-7
    return -np.sum(np.log(y[np.arange(batch_size),t]+delta))/batch_size