from MNIST.mnist import mnist
from MNIST.neural_network import neural_network


def mnist_main():
    mnist_entity=mnist()
    mnist_entity.init_mnist()
    #(train_img,train_label),(test_img,test_label)=mnist_entity.load_mnist(normalize=False,flatten=True)

    mnist_network=neural_network()
    mnist_network.test_accuracy(100)

if __name__ == "__main__":
    mnist_main()