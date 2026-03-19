from config import config
import os.path
import gzip
import pickle
import os
import numpy as np
from PIL import Image

class mnist():
    pkl_buffer = config.DL_MNIST_DATA_PATH / "mnist.pkl"
    train_num = 60000
    test_num = 10000
    img_dim = (1, 28, 28)
    img_size = 784
            
    def _load_label(self, label_path):
        with gzip.open(label_path, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
        return labels

    def _load_img(self, img_path):   
        with gzip.open(img_path, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, self.img_size)        
        return data
    
    def _convert_numpy(self):
        dataset = {}
        dataset['train_img'] =  self._load_img(config.MNIST_FILE_PATH['TRAIN_IMG'])
        dataset['train_label'] = self._load_label(config.MNIST_FILE_PATH['TRAIN_LABEL'])    
        dataset['test_img'] = self._load_img(config.MNIST_FILE_PATH['TEST_IMG'])
        dataset['test_label'] = self._load_label(config.MNIST_FILE_PATH['TEST_LABEL'])
        
        return dataset

    def init_mnist(self):
        dataset = self._convert_numpy()
        print("Creating pickle file ...")
        with open(self.pkl_buffer, 'wb') as f:
            pickle.dump(dataset, f, -1)
        print("Done!")

    def _change_one_hot_label(X):
        T = np.zeros((X.size, 10))
        for idx, row in enumerate(T):
            row[X[idx]] = 1
            
        return T
    

    def load_mnist(self, normalize=True, flatten=True, one_hot_label=False):
        """load MNIST data
        
        Parameters
        ----------
        normalize : normalize the image data value to 0.0~1.0
        one_hot_label : 
            true then retrun label as one-hot array
            one-hot array means array like [0,0,1,0,0,0,0,0,0,0]
        flatten : if flatten the image data to one dimension
        
        Returns
        -------
        (train_img, train_label), (test_img, test_label)
        """
        if not os.path.exists(self.pkl_buffer):
            self.init_mnist()
            
        with open(self.pkl_buffer, 'rb') as f:
            dataset = pickle.load(f)
        
        if normalize:
            for key in ('train_img', 'test_img'):
                dataset[key] = dataset[key].astype(np.float32)
                dataset[key] /= 255.0
                
        if one_hot_label:
            dataset['train_label'] = self._change_one_hot_label(dataset['train_label'])
            dataset['test_label'] = self._change_one_hot_label(dataset['test_label'])
        
        if not flatten:
            for key in ('train_img', 'test_img'):
                dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

        return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label']) 
    
    def img_show(self,img):
        pil_img=Image.fromarray(np.uint8(img))
        pil_img.show()

    def init_network():
        with open("sample_weight.pkl", 'rb') as f:
            network = pickle.load(f)
        return network
    
    def predict(network, x):
        W1, W2, W3 = network['W1'], network['W2'], network['W3']
        b1, b2, b3 = network['b1'], network['b2'], network['b3']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = sigmoid(a2)
        a3 = np.dot(z2, W3) + b3
        y = softmax(a3)

        return y