import numpy as np 
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

def linear_model():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.random_normal([784, 10]))
    b = tf.Variable(tf.zeros([10])+0.1)

    y = tf.nn.softmax(tf.matmul(x, W) + b)
    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(100):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            # 执行步骤，但是需要填入placeholder的值
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys}) 
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
        print("linear_model over!")

def unit_test():
    linear_model()

def main():
    unit_test()

if __name__ == "__main__":
    main()