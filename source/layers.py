import numpy as np
class Sigmoid():
    def __init__(self):
        self.params=[]
    
    def forward(self,x):
        return 1/(1+np.exp(-x))
    
class Relu():
    def __init__(self):
        self.params=[]

    def forward(self,x):
        return np.maximum(0,x)
    
class Softmax():
    def __init__(self):
        self.params=[]

    def forward(self,x):
        if x.ndim == 2:
            x = x.T
            x = x - np.max(x, axis=0)
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            return y.T
        x = x - np.max(x)
        return np.exp(x) / np.sum(np.exp(x))
    
class Step():
    def __init__(self):
        self.params=[]

    def forward(self,x):
        return np.array(x > 0, dtype=np.int)

class Affine():
    def __init__(self,W,B):
        self.params=[W,B]

    def forward(self,x):
        W,b=self.params
        return np.dot(x,W)+b
    
class CrossEntropyError():
    def __init__(self):
        self.params=[]

    def forward(self,x,t):
        if x.ndim==1:
            x.reshape(1,x.size)
            t.reshape(1,x.size)

        delta=1e-7
        batch_size = x.shape[0]
        return -np.sum(np.log(x[np.arange(batch_size),t]+delta))/batch_size
        