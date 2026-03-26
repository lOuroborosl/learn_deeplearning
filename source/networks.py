import layers
import numpy as np
class TwoLayerNet():
    def __init__(self,input_size,hidden_size,output_size):
        I,H,O=input_size,hidden_size,output_size
        self.loss=0

        W1=np.random.uniform(low=-np.sqrt(6/(I+H)),high=np.sqrt(6/(I+H)),size=(I,H))
        b1=np.zeros(H)

        W2=np.random.uniform(low=-np.sqrt(6/(H+O)),high=np.sqrt(6/(H+O)),size=(H,O))
        b2=np.zeros(O)

        self.layers=[
            layers.Affine(W1,b1),
            layers.Sigmoid(),
            layers.Affine(W2,b2),
            layers.Softmax()
        ]
        
        self.params=[]
        for layer in self.layers:
            self.params+=layer.params

    def predict(self,x):
        for layer in self.layers:
            x=layer.forward(x)
        return x
    
    def getLoss(self,x,t):
        y=self.predict(x)
        loss=layers.CrossEntropyError()
        self.loss=loss.forward(y,t)
        return self.loss