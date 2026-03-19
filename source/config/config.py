from pathlib import Path

DL_CONFIG_PATH     =    Path(__file__).resolve().parent
DL_SOURCE_PATH     =    Path(DL_CONFIG_PATH).resolve().parent
DL_BASE_PATH       =    Path(DL_SOURCE_PATH).resolve().parent
DL_DATA_PATH       =    DL_BASE_PATH    /   "data"
DL_IMG_DATA_PATH   =    DL_BASE_PATH    /   "data"    / "image"
DL_MNIST_DATA_PATH =    DL_DATA_PATH    /   "mnist"
MNIST_FILE_PATH    =    {
    "TRAIN_IMG"    :    DL_MNIST_DATA_PATH / "train-images-idx3-ubyte.gz",
    "TRAIN_LABEL"  :    DL_MNIST_DATA_PATH / "train-labels-idx1-ubyte.gz",
    "TEST_IMG"     :    DL_MNIST_DATA_PATH / "t10k-images-idx3-ubyte.gz",
    "TEST_LABEL"   :    DL_MNIST_DATA_PATH / "t10k-labels-idx1-ubyte.gz"
}
WEIGHT_FILE_PATH   =    DL_MNIST_DATA_PATH / "sample_weight.pkl"