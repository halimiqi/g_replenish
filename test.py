import tensorflow as tf
import numpy as np
import utils

def main():
    A,X,z = utils.load_npz("data/citeseer.npz")
    A,X,z = utils.load_npz("data/cora.npz")

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
main()

