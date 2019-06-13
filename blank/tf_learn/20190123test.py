import numpy as np 
import tensorflow as tf 
import pandas as pd 
import time 

def unit_test5():
    a = time.clock()
    for i in range(2000):
        print(i)
    b = time.clock()
    print("程序运行时间:", b-a,"s")

def unit_test4():
    a = tf.Variable(tf.constant([[1,2],[3,4],[5,6]]))
    b = tf.split(a, 3, 0)

    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        a = sess.run(a)
        print(a, a.shape)
        b = sess.run(b)
        print(b, type(b))
    print("unit_test4 over!")
    for i in range(len(b)):
        print(i, b[i], type(b[i]))


def unit_test3():
    a = tf.Variable(tf.random_normal([2,2]))
    # 测试函数transpose的作用 
    b = tf.transpose(a, [1,0])

    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        a = sess.run(a)
        b = sess.run(b)
        print(a, a.shape)
        print(b, b.shape)
        print("unit_test3 over!")


def unit_test2():
    x = tf.Variable(tf.ones([2,3]))
    w = tf.Variable(tf.ones([3,1])*3)

    y1 = tf.matmul(x, w)
    a = tf.constant([1.0])
    y2 = y1 + a
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        x1 = sess.run(x)
        print(x1,x1.shape)
        w1 = sess.run(w)
        print(w1,w1.shape)
        y1_ = sess.run(y1)
        print(y1_, y1_.shape)
        y2_ = sess.run(y2)
        print(y2_, y2_.shape)
        print("unit_test2 over")


def unit_test1():
    weight = tf.Variable(tf.random_normal([2,2]))
    data = tf.ones([1,2])
    y = tf.matmul(data, weight)
    y_ = tf.nn.relu(y)
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)
        wi = sess.run(weight)
        print(wi)
        data = sess.run(data)
        print(data)
        y = sess.run(y)
        print(y, y.shape)
        y_ = sess.run(y_)
        print(y_, y_.shape)
        
    print("session over")

def main():
    unit_test5()

if __name__ == "__main__":
    main()
