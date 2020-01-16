import tensorflow as tf
import numpy as np
import utils

def main():
    A,X,z = utils.load_npz("data/citeseer.npz")
    A,X,z = utils.load_npz("data/cora.npz")

    return

def test():
    a = 0
    new_adj_for_del_softmax = tf.constant([5,3,4,6,7,3,2,1,0,0,0,1,4,2,3], dtype = tf.float32)
    del_gather_idx = tf.where(new_adj_for_del_softmax > 0)
    new_adj_del_softmax_gather = tf.gather(new_adj_for_del_softmax, del_gather_idx[:,0])
    new_indexes_gather = tf.multinomial(tf.log([new_adj_del_softmax_gather]), 5)  # this is the sample section
    new_indexes = tf.gather(del_gather_idx[:,0], new_indexes_gather[0])

    with tf.Session() as sess:
        test2 = sess.run(new_indexes)
        print(test2)
    return
test()

