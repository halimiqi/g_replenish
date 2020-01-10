import tensorflow as tf
import numpy as np
import utils
from layers import FullyConnect


def main():
    A,X,z = utils.load_npz("data/citeseer.npz")
    A,X,z = utils.load_npz("data/cora.npz")

    return

def test():
    a = 0
    a += tf.ones([5,5])
    c = tf.diag_part(a)
    d = a + 1
    per = tf.random_normal([5])
    op = tf.train.RMSPropOptimizer(learning_rate = 0.01)
    Z_tilde = FullyConnect(output_size=5, scope="flip_weight")(d)
    feature_reg = tf.matmul(tf.matmul(Z_tilde, a, transpose_a=True), Z_tilde)
    feature_diag = tf.diag_part(feature_reg)
    new_indexes = tf.multinomial(tf.log([feature_diag]), 5)  # this is the sample section
    per_feature = tf.reduce_sum(tf.log(tf.gather(feature_diag, new_indexes[0])))
    loss = per_feature * per
    grad = tf.gradients(loss, d)
    with tf.Session() as sess:
        test2 = sess.run(loss)
        test1 = sess.run(grad)
        print(test1)
        print(test2)

test()
