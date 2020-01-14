import tensorflow as tf
import numpy as np
import utils

def main():
    A,X,z = utils.load_npz("data/citeseer.npz")
    A,X,z = utils.load_npz("data/cora.npz")

    return

def test():
    a = 0
    a = np.arange(9)
    a = a.reshape([3,3])
    a = tf.constant(a)
    b = tf.one_hot(4,9, dtype=tf.int32)
    c = tf.reshape(b, [3,3])


    with tf.Session() as sess:
        test2 = sess.run(b)
        test3 = sess.run(c)
        print(test2)
    return
test()

