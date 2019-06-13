import matplotlib
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
from tensorflow.examples.tutorials.mnist import input_data

def load_data():
    mnist = input_data.read_data_sets("mnist/", one_hot = True, reshape = False)
    print(type(mnist))
    print('train image shape:', mnist.train.images.shape, type(mnist.train.images))
    print('trian label shape:', mnist.train.labels.shape)
    print('val image shape:', mnist.validation.images.shape)
    print('test image shape:', mnist.test.images.shape)
    print('test label shape:', mnist.test.labels.shape)
    print("load_data over")
    train_images = mnist.train.images
    train_labels = mnist.train.labels
    test_images = mnist.test.images
    test_labels = mnist.test.labels
    return train_images, train_labels, test_images, test_labels

def plot_image(a):
    """
    DESC: 输入一个图像矩阵，画出图像
    """
    image = a.reshape(28, 28)
    plt.imshow(image, cmap=matplotlib.cm.binary, interpolation="nearest")
    plt.axis("off")
    plt.show()

def model_def():
    '''
    x = tf.Variable(tf.random_normal([1,3,3,5]))
    filter_ar = tf.Variable(tf.random_normal([1,1,5,1]))
    op = tf.nn.conv2d(x, filter_ar, strides=[1,1,1,1], padding='VALID')
    '''
    mnist = input_data.read_data_sets("mnist/", one_hot = True)
    #train_images = mnist.train.images
    #train_labels = mnist.train.labels
    #test_images = mnist.test.images
    #test_labels = mnist.test.labels

    x_ = tf.placeholder(tf.float32, [None, 784])
    x = tf.reshape(x_, [-1,28,28,1])         #转换输入数据shape,以便于用于网络中
    y_actual = tf.placeholder(tf.float32, shape=[None, 10])
    # 第一层卷积层+池化层
    w_conv1 = weight_var([5,5,1,32])
    b_conv1 = bias_var([32])
    h_conv1 = tf.nn.relu(conv2d(x, w_conv1) + b_conv1)
    h_pool1 = max_pool(h_conv1)

    # 第二层卷积层+池化层
    w_conv2 = weight_var([5,5,32,64])
    b_conv2 = bias_var([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
    h_pool2 = max_pool(h_conv2)

    # 全连接层
    w_fc1 = weight_var([7*7*64, 1024])
    b_fc1 = bias_var([1024])
    h_pool2 = tf.reshape(h_pool2, shape=[-1,7*7*64])
    h_fc1 =tf.nn.relu(tf.matmul(h_pool2, w_fc1) + b_fc1)

    # drop功能
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # softmax层
    W_fc2 = weight_var([1024, 10])
    b_fc2 = bias_var([10])
    y_predict=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)  

    cross_entropy = -tf.reduce_sum(y_actual*tf.log(y_predict))     #交叉熵
    train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)    #梯度下降法
    correct_prediction = tf.equal(tf.argmax(y_predict,1), tf.argmax(y_actual,1))    
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))                 #精确度计算
    sess=tf.InteractiveSession()                          
    sess.run(tf.initialize_all_variables())
    for i in range(20000):
        #sess.run(tf.initialize_all_variables())
        batch = mnist.train.next_batch(50)
        if i%100 == 0:                  #训练100次，验证一次
            train_acc = accuracy.eval(feed_dict={x_:batch[0], y_actual: batch[1], keep_prob: 1.0})
            print ('step %d, training accuracy %g'%(i,train_acc))
            train_step.run(feed_dict={x_: batch[0], y_actual: batch[1], keep_prob: 0.5})

    test_acc=accuracy.eval(feed_dict={x_: mnist.test.images, y_actual: mnist.test.labels, keep_prob: 1.0})
    print ("test accuracy %g"%test_acc) 

    '''
    var_init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(var_init)
        y = sess.run(op)
        print(y)
    '''

def weight_var(shape):
    weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    return weight

def bias_var(shape):
    bias = tf.Variable(tf.constant(0.1, shape=shape))
    return bias

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')

def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')



def test_cov():
    x = tf.Variable(tf.ones([1,3,3,5]))
    y = tf.Variable(tf.ones([2,2,5,2]))
    init_var = tf.global_variables_initializer()
    op = tf.nn.conv2d(x, y, strides=[1,1,1,1], padding="VALID")
    with tf.Session() as sess:
        sess.run(init_var)
        z = sess.run(op)
        print(z, z.shape)
    print("test_cov over")


def unit_test():
    train_images, train_labels, test_images, test_labels = load_data()
    plot_image(train_images[2])
    print("unit_test over")

def main():
    #load_data()
    model_def()
    #test_cov()

if __name__ == "__main__":
    main()
    print("main thread over")
    

