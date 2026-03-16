from config import config

import numpy as np

import matplotlib
matplotlib.use('QtAgg')

from matplotlib import pyplot as plt
plt.ion()

from matplotlib.image import imread



def np_test():
    x=np.array([[0,1,2],[2,3,4]])
    print("x is")
    print(x)
    print("x shape is")
    print(x.shape)

    y=np.array([1,2,3])
    print("y is")
    print(y)
    print("x*y=")
    print(x*y)
    print("x-y=")
    print(x-y)


def plt_test():
    x=np.arange(0,6,0.1)
    y=np.sin(x)
    y2=np.cos(x)
    plt.plot(x,y,label="sin")
    plt.plot(x,y2,linestyle="--",label="cos")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

def imread_test():
    img=imread(config.DL_IMG_DATA_PATH / "lena.png")
    plt.imshow(img)
