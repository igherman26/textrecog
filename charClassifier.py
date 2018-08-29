import tflearn, sys
import numpy as np
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels
from tflearn.data_utils import shuffle, to_categorical

CHARS = [   "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
            "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
            "u", "v", "w", "x", "y", "z", "A", "B", "C", "D",
            "E", "F", "G", "H", "I", "J", "K", "L", "M", "O",
            "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X,",
            "Y", "Z"]

TRAIN_IMAGES_PATH = "./data/emnist/emnist-byclass-train-images-idx3-ubyte.gz"
TRAIN_LABELS_PATH = "./data/emnist/emnist-byclass-train-labels-idx1-ubyte.gz"
TEST_IMAGES_PATH = "./data/emnist/emnist-byclass-test-images-idx3-ubyte.gz"
TEST_LABELS_PATH = "./data/emnist/emnist-byclass-test-labels-idx1-ubyte.gz"

MODEL_PATH = "./model/char-classifier.tfl"

class Classifier:
    def __init__(self):
        self.model = loadModelData()[0]
        self.model.load(MODEL_PATH)

    # returns the classified character
    def classify(self, X):
        prediction = np.argmax(self.model.predict(X))

        #print ("prediction {}".format(CHARS[prediction]))

        # return the prediction as the max index (of prediction list) entry in CHARS
        return CHARS[prediction] 

def loadModelData():
    # load EMNIST data
    with open(TRAIN_IMAGES_PATH, 'rb') as f:
        train_images = extract_images(f)
    with open(TRAIN_LABELS_PATH, 'rb') as f:
        train_labels = extract_labels(f)

    with open(TEST_IMAGES_PATH, 'rb') as f:
        test_images = extract_images(f)
    with open(TEST_LABELS_PATH, 'rb') as f:
        test_labels = extract_labels(f)

    # "rename" to make it similar to the tutorial
    # https://github.com/tflearn/tflearn/blob/master/examples/images/convnet_mnist.py
    X, Y, testX, testY = train_images, train_labels, test_images, test_labels

    # data preprocessing
    X = X.reshape([-1, 28, 28, 1])
    testX = testX.reshape([-1, 28, 28, 1])
    Y = to_categorical(Y, nb_classes=62)
    testY = to_categorical(testY, nb_classes=62)

    # Building convolutional network
    # the input is a 28x28 image with 1 channel
    network = input_data(shape=[None, 28, 28, 1], name='input')
    
    # 3 x convolution + max pooling
    network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
    network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
    network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)

    # fully connected with 512 nodes + some dropout
    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.5)
    # fully connected with 62 nodes which are the outputs
    network = fully_connected(network, 62, activation='softmax')

    # train the network with regression
    network = regression(network, optimizer='adam',
                         loss='categorical_crossentropy', name='target')

    # Training
    model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path='classifier.tfl.ckpt')

    return model, X, Y, testX, testY

def trainModel(model, X, Y, testX, testY, path, epochs=5):
    model.fit({'input': X}, {'target': Y}, n_epoch=epochs, batch_size=196,
               validation_set=({'input': testX}, {'target': testY}),
               snapshot_step=100000, show_metric=True, run_id='convnet_emnist', snapshot_epoch=True)

    model.save(path)

def main():
    if len(sys.argv) > 2:
        print ("usage: {} [model_path]".format(sys.argv[0]))
        return

    # load model data (it's structure)
    model, X, Y, testX, testY = loadModelData()

    # resume model training
    if len(sys.argv) == 2:
        model.load(sys.argv[1])
        trainModel(model, X, Y, testX, testY, sys.argv[1])
    else:
        model.load(MODEL_PATH)
        trainModel(model, X, Y, testX, testY, MODEL_PATH)


if __name__ == "__main__":
    main()