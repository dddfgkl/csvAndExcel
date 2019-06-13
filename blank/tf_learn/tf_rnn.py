import numpy as np 
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

def compare_difftype_mnist():
    """
        两种读取方式本质是相同的
    """
    mnist_type_one = input_data.read_data_sets("MNIST_data/", one_hot = True)
    mnist_type_two = input_data.read_data_sets("minist/", one_hot = True)
    type_two_train_images = mnist_type_one.train.images
    type_two_train_labels = mnist_type_one.train.labels
    type_one_train_images = mnist_type_two.train.images
    type_one_train_labels = mnist_type_two.train.labels
    print("MNIST images: ", type_one_train_images.shape)
    print("MNIST labels: ", type_one_train_labels.shape)
    print("mnist images: ", type_two_train_images.shape)
    print("mnist labels: ", type_two_train_labels.shape)
    
def unit_test():
    compare_difftype_mnist()

def main():
    unit_test()

if __name__ == "__main__":
    main()
    print("main thread over")