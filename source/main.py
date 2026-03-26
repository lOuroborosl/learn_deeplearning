from mnist import mnist
import neural_network
import layers
import numpy as np
from networks import TwoLayerNet


def main():
    mnist_entity=mnist()
    mnist_entity.init_mnist()
    (train_img,train_label),(test_img,test_label)=mnist_entity.load_mnist(normalize=False,flatten=True)
    network=TwoLayerNet(784,784,10)
    print(network.getLoss(test_img,test_label))
    print(train_label[5])
    #mnist_network=neural_network()
    #mnist_network.test_accuracy(100)

if __name__ == "__main__":
    main()