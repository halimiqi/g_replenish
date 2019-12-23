import tensorflow as tf
import numpy as np


def main():

    a = tf.random_uniform(
        (1,5), minval=-0.5,
        maxval=0.5, dtype=tf.float32)
    a = tf.reshape(a, [-1])
    b = tf.nn.top_k(a, k = 2)
    mask = tf.reshape(tf.ones([1,5]),[-1])
    for i in range(2):
        mask_onehot = tf.one_hot(b[1][i], mask.shape[0], dtype=tf.float32)
        mask = mask - mask_onehot


    with tf.Session() as sess:
        for i in range(5):
            out_b = sess.run(mask)
            #print(out_a)
            print(out_b)
    return

def test():
    a = 0
    a += tf.ones([5])
    b = a + 1
    with tf.Session() as sess:
        test1 = sess.run(b)
        test2 = sess.run(b)
        print(test1)
        print(test2)
    return
#main()
test()
