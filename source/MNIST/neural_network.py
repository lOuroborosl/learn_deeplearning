import numpy as np
import pickle
from MNIST import mnist
from MNIST.common.functions import sigmoid, softmax
import config.config as config

class neural_network(object):
    def get_data(self):
        mnist_entity=mnist.mnist()
        (x_train, t_train), (x_test, t_test) = mnist_entity.load_mnist(normalize=True, flatten=True, one_hot_label=False)
        return x_test, t_test


    def init_network(self):
        with open(config.WEIGHT_FILE_PATH, 'rb') as f:
            network = pickle.load(f)
        return network


    def predict(self,network, x):
        W1, W2, W3 = network['W1'], network['W2'], network['W3']
        b1, b2, b3 = network['b1'], network['b2'], network['b3']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = sigmoid(a2)
        a3 = np.dot(z2, W3) + b3
        y = softmax(a3)

        return y

    def test_accuracy(self,batch_size):
        x, t = self.get_data()
        network = self.init_network()
        accuracy_cnt = 0
        for i in range(0,len(x),batch_size):
            x_batch=x[i:i+batch_size]
            y_batch = self.predict(network, x_batch)
            p= np.argmax(y_batch,axis=1) # the highest value
            accuracy_cnt += np.sum(p==t[i:i+batch_size])

        print("Accuracy:" + str(float(accuracy_cnt) / len(x)))