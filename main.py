import numpy as np
import tensorflow as tf
from utils import load_CIFAR10, get_data_batch

init = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)
reg = tf.contrib.layers.l2_regularizer(0.1)


def double_conv(x, c1, c2, z1, z2, layer_name):
    N, H, W, _ = x.get_shape().as_list()  # as_list() is very important
    with tf.variable_scope('weight', initializer=init, regularizer=reg):
        w_meta = tf.get_variable('w_meta_'.join(layer_name), [c2, z2, z2, c1])
    I1 = tf.constant(np.eye(c1 * z1**2).reshape(z1, z1, c1, c1 * z1**2), dtype=tf.float32)  # filter
    w_mid = tf.nn.conv2d(w_meta, I1, strides=[1, 1, 1, 1], padding='VALID')  # [c2, z2-z1+1, z2-z1+1, c1 * z1**2]
    w_mid = tf.reshape(w_mid, [c2 * (z2-z1+1)**2, -1])
    w_mid = tf.transpose(w_mid)
    w_mid = tf.reshape(w_mid, [z1, z1, c1, -1])  # Reorganize to [z1, z1, c1, c2 * (z2-z1+1)**2]
    O = tf.nn.conv2d(x, w_mid, strides=[1, 1, 1, 1], padding='SAME')  # [N, H, W, c2 * (z2-z1+1)**2]
    O = tf.reshape(O, [N, H, W, c2, z2-z1+1, z2-z1+1])  # [N, H, W, c2, (z2-z1+1)**2]
    O = tf.transpose(O, [0, 4, 5, 3, 1, 2])  # [N, z2-z1+1, z2-z1+1, c2, H, W]
    O = tf.reshape(O, [N, z2-z1+1, z2-z1+1, -1])
    I2 = tf.nn.max_pool(O, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')  # [N, (z2-z1+1)/2, (z2-z1+1)/2, c2*H*W]
    I2 = tf.reshape(I2, [N, ((z2-z1+1)/2)**2, c2, H, W])
    I2 = tf.transpose(I2, [0, 3, 4, 1, 2])
    I2 = tf.reshape(I2, [N, H, W, -1])
    with tf.variable_scope('bias', initializer=tf.constant_initializer(0.)):
        b = tf.get_variable('b_conv_'.join(layer_name), shape=I2.get_shape()[-1])
    out = I2 + b
    return out


def model(x):
    with tf.variable_scope('weight', initializer=init, regularizer=reg):
        w_fc_1 = tf.get_variable('w_fc_1', [8 * 8 * 64, 1000])
        w_fc_2 = tf.get_variable('w_fc_2', [1000, 10])
    with tf.variable_scope('bias', initializer=tf.constant_initializer(0.)):
        b_fc_1 = tf.get_variable('b_fc_1', [1000])
        b_fc_2 = tf.get_variable('b_fc_2', [10])
    a_dconv_1 = double_conv(x, 3, 32, 3, 4, '1')
    h_dconv_1 = tf.nn.elu(a_dconv_1)
    pool_1 = tf.nn.max_pool(h_dconv_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    a_dconv_2 = double_conv(pool_1, 32, 64, 3, 4, '2')
    h_dconv_2 = tf.nn.elu(a_dconv_2)
    pool_2 = tf.nn.max_pool(h_dconv_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    pool_2 = tf.reshape(pool_2, [-1, 8 * 8 * 64])
    a_fc_1 = tf.matmul(pool_2, w_fc_1) + b_fc_1
    h_fc_1 = tf.nn.elu(a_fc_1)
    logit = tf.matmul(h_fc_1, w_fc_2) + b_fc_2
    prob = tf.nn.softmax(logit)
    return prob


if __name__ == '__main__':
    X_train, y_train = load_CIFAR10('cifar-10-batches-py', load_test=False)

    X = tf.placeholder(tf.float32, shape=[64, 32, 32, 3])
    y = tf.placeholder(tf.float32, shape=[64, 10])

    prob = model(X)
    ce = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prob), reduction_indices=[1]))

    train_step = tf.train.MomentumOptimizer(5e-3, 0.95).minimize(ce)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prob, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in xrange(500):
            X_batch, y_batch = get_data_batch(X_train, y_train, 64)
            train_step.run(feed_dict={X: X_batch, y: y_batch})
            if i % 10 == 0:
                acc = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
                print acc
