import tensorflow as tf 
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np 


def load_data():
    """
    DESC: 对mnist数据集进行处理，包括画出来
    """
    mni = load_digits()
    print(mni.data.shape)
    print(mni.target.shape)
    x = mni.data
    y = mni.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    '''
    a = x[1]
    image = a.reshape(8, 8)
    plt.imshow(image, cmap=matplotlib.cm.binary, interpolation="nearest")
    plt.axis("off")
    plt.show()
    '''
    print(x_train.shape, x_test.shape)
    print(y_train.shape, y_test.shape)
    #print(y)
    #enc = OneHotEncoder()
    #enc.fit(y_test)
    #ans = enc.fit_transform(y_test)
    #print(ans.toarray())
    return x_train, x_test, y_train, y_test

def model_def():
    # tf.Variable()是一个类
    #正确率一直都只有0.1，哪里有错吗
    x_train, x_test, y_train, y_test = load_data()
    y_test = one_hot_process(y_test)
    y_train = one_hot_process(y_train)
    x = tf.placeholder(dtype=tf.float32, shape=[None, 64])
    w = tf.Variable(tf.zeros(shape=[64, 10]))
    b = tf.Variable(tf.zeros(shape=[10]))
    y = tf.nn.softmax(tf.matmul(x, w) + b)
    y_ = tf.placeholder(dtype=tf.float32, shape=[None, 10])
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    init = tf.global_variables_initializer()
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    #correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))
    with tf.Session() as sess:
        sess.run(init)
        for _ in range(10000):
            sess.run(train_step, feed_dict={x:x_train, y_:y_train})
        # tf.equal()方法对两个相同维度的矩阵对应元素进行比较，若相等，该位置为true
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))
        acc = sess.run(accuracy, feed_dict={x:x_train, y_:y_train})
        print(acc)

def one_hot_process(lable):
    # 对lable进行one-hot编码
    a = lable.reshape(-1,1)
    #print(a[:10,:])
    enc = OneHotEncoder()
    #enc.fit(a)
    ans = enc.fit_transform(a)
    #print(ans[:10,:].toarray())
    return ans.toarray()

def unit_test():
    ml_sample3()

def ml_sample2():
    mnist = fetch_mldata('MNIST original')
    x = mnist['data']
    y = mnist['target']
    print(x.shape, y.shape)
    print("ml_sample2 over!")

# tf.equal()方法使用例子，返回值是一个array
def ml_sample3():
    matrix1 = tf.constant([1,2,3,4])
    matrix2 = tf.constant([1,2,3,5])
    equal_mt = tf.equal(matrix1,matrix2)
    with tf.Session() as sess:
        mt = sess.run(equal_mt)
        print(mt, type(mt))
    print("ml_sample3 over")

def main():
    #unit_test()
    model_def()

if __name__ == "__main__":
    main()
