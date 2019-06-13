import tensorflow as tf 
import numpy as np 

"""
    tensorflow模型构建主要有两部分组成
        1、构建图
        2、在会话中启动一个图
"""
def simple_sample1():
    hello = tf.constant('hello world')
    with tf.Session() as sess:
         print(sess.run(hello))
         #sess.close()
         # 显示的colse在with代码块中并没有必要，with代码块结束的时候会自动调用close对象关闭sess
    a = tf.constant(100)
    b = tf.constant(200)
    with tf.Session() as sess:
        print(sess.run(a+b))
        #sess.close()
    print("simple_sample1 over")

def simple_sample2():
    matrix1 = tf.constant([[1.,1.],[1.,1.]])
    matrix2 = tf.constant([[2.,2.],[2.,2.]])
    matrix3 = matrix1 + matrix2
    matrix4 = matrix1 - matrix2
    # 对于array对象来说，*并不是数学定义上的矩阵相乘，仅仅是数量积
    # tensor本质是array对象，所以按照array定义的方式运行
    matrix5 = matrix1 * matrix2
    # matrix按照数学定义的矩阵乘法来运算需要用到函数matmul()
    matrix6 = tf.matmul(matrix1, matrix2)
    with tf.Session() as sess:
        print(sess.run(matrix3))
        print(sess.run(matrix4))
        print(sess.run(matrix5))
        print(sess.run(matrix6))
        #sess.close()
    print("simple_sample2 over!")

def simple_sample3():
    matrix1 = tf.constant([1.,2.,3.])
    a = matrix1.shape
    b = matrix1.get_shape()
    c = tf.shape(matrix1)

    a = a.as_list()
    b = b.as_list()
    #c = c.as_list()
    print(a)
    print(b)
    with tf.Session() as sess:
        d = sess.run(c)
        print(d)

def simple_sample4():
    a = tf.constant(1.0)
    b = tf.constant(2.0)
    c = tf.constant(3.0)
    intermd = tf.add(a,b)
    mul = tf.multiply(c, intermd)
    with tf.Session() as sess:
        # 可以一次性取回多个tensor
        ret = sess.run([intermd, mul])
        print(ret)
        print(type(ret))
    print("simple_sample4 over!")


def simple_sample5():
    input1 = tf.placeholder(dtype=tf.float32)
    input2 = tf.placeholder(dtype=tf.float32)   
    output = tf.multiply(input1,input2)
    with tf.Session() as sess:
        c = sess.run([output], feed_dict={input1:[4],input2:[6.]})
        print(c)
        print(type(c))
    print("simple_sample5 over!")

def simple_sample6():
    a = np.array([[1,1,1],[2,2,2]])
    b = np.array([1,1,1])
    c = np.zeros(10)
    print(c, c.shape)
    c = a + b
    print(c)

def main():
    simple_sample6()

if __name__ == "__main__":
    main()

