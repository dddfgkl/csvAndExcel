from tensorflow.examples.tutorials.mnist import input_data 
import tensorflow as tf 
import numpy as np 
# 在这里做数据加载，还是使用那个MNIST的数据，
# 以one_hot的方式加载数据，记得目录可以改成之前已经下载完成的目录 
mnist = input_data.read_data_sets("mnist", one_hot=True)

# RNN使用到的参数
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

# 神经网络参数
n_input = 28  # 输入层的n
n_steps = 28  # 28长度
n_hidden = 128  # 隐含层的特征数
n_classes = 10  # 输出的数量，因为是分类问题，0~9个数字，这里一共有10个

x = tf.placeholder("float", [None, n_steps, n_input])
istate = tf.placeholder("float", [None, 2 * n_hidden])
y = tf.placeholder("float", [None, n_classes])

weights = { 'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])), # Hidden layer weights 
            'out': tf.Variable(tf.random_normal([n_hidden, n_classes])) }

biases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def RNN(_X, _istate, _weights, _biases):
    # 规整输入的数据
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size

    _X = tf.reshape(_X, [-1, n_input])  # (n_steps*batch_size, n_input)
    _X = tf.matmul(_X, _weights['hidden']) + _biases['hidden']
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    _X = tf.split(_X, n_steps, 0)  # n_steps * (batch_size, n_hidden)
    outputs, states = tf.nn.static_rnn(lstm_cell, _X, dtype=tf.float32)
    return tf.matmul(outputs[-1], _weights['out']) + _biases['out']

pred = RNN(x, istate, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)) # Softmax loss 
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) 

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)
step = 1
while step*batch_size < training_iters:
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    batch_xs = batch_xs.reshape((batch_size, n_steps, n_input))
    sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, 
            istate: np.zeros((batch_size, 2 * n_hidden))})

    if step % display_step == 0:
        acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, 
                istate: np.zeros((batch_size, 2 * n_hidden))})
        loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, 
                istate: np.zeros((batch_size, 2 * n_hidden))})
        print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + 
                ", Training Accuracy= " + "{:.5f}".format(acc))
    step += 1
print("Optimization Finished!")
test_len = 256 
test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input)) 
test_label = mnist.test.labels[:test_len] 
print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label, istate: np.zeros((test_len, 2 * n_hidden))}))

