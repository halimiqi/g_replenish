import numpy as np
import scipy
import tensorflow as tf
from layers import GraphConvolution, GraphConvolutionSparse,InnerProductDecoder, FullyConnect, Graphite, \
    GraphiteSparse,Scale,Dense,GraphiteSparse_simple, Graphite_simple,GraphConvolutionSparse_denseadj, GraphConvolution_denseadj
from ops import batch_normal
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from optimizer import Optimizergaegan
flags = tf.app.flags
FLAGS = flags.FLAGS


class gaegan(object):

    def __init__(self,placeholders, num_features,num_nodes, features_nonzero,learning_rate_init ,if_drop_edge = True, **kwargs):
        # processing the name and the logging
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.input_dim = num_features
        self.inputs = placeholders['features']
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.adj_ori = placeholders['adj_orig']
        #self.node_labels = placeholders['node_labels']
        self.features_nonzero = features_nonzero
        self.batch_size = FLAGS.batch_size
        self.latent_dim = FLAGS.latent_dim
        self.n_samples = num_nodes  # this is the number of nodes in the nodes
        # preprocess the dataset and define the parameter we need
        self.zp = tf.random_normal(shape=[self.n_samples, self.latent_dim])
        self.learning_rate_init = learning_rate_init
        self.if_drop_edge = if_drop_edge
        return

    def build_model(self):
        #the learning rate decay part
        # build the model
        self.z_x = self.encoder(self.inputs)
        self.x_tilde = 0
        self.adj_dense = tf.sparse_tensor_to_dense(self.adj, default_value=0, validate_indices=False, name=None) # normalize adj
        self.new_adj_output = self.adj_dense
        self.adj_ori_dense = tf.sparse_tensor_to_dense(self.adj_ori, default_value=0, validate_indices=False, name=None) #A + I
        if FLAGS.generator == "inner_product":
            self.x_tilde = self.generate(self.z_x, self.z_x.shape[1])
        if FLAGS.generator == "graphite":
            self.x_tilde = self.generate_graphite(self.z_x, self.z_x.shape[1], self.input_dim)
        if FLAGS.generator == "graphite_attention":
            self.x_tilde = self.generate_graphite_simple_no_innerprod(self.z_x, self.z_x.shape[1], self.input_dim)
        if FLAGS.generator == "dense_attention":
            self.x_tilde = self.generate_attention(self.z_x, self.z_x.shape[1], self.input_dim)
        if FLAGS.generator == "dense":
            self.x_tilde = self.generate_dense(self.z_x, self.z_x.shape[1], self.input_dim)
            self.x_tilde_output_ori = self.x_tilde
        if self.if_drop_edge != False:
            #self.x_tilde, self.new_adj_output = self.delete_k_edge(self.x_tilde, self.adj_ori_dense)  # convert the original graph with k edges
            #self.x_tilde = np.zeros
            for time in range(FLAGS.delete_edge_times):
                self.x_tilde_out, self.new_adj_output = self.delete_k_edge_min_new(self.x_tilde, self.adj_ori_dense, k = FLAGS.k)
                if time == 0:
                    self.x_tilde_average = self.x_tilde_out
                else:
                    self.x_tilde_average = self.x_tilde_average + self.x_tilde_out
            self.x_tilde_average = self.x_tilde_average / FLAGS.delete_edge_times
            #self.x_tilde, self.new_adj_output = self.delete_k_edge_max(self.x_tilde, self.adj_ori_dense, k = FLAGS.k)
            #self.x_tilde_deleted = self.x_tilde_out
            self.x_tilde_deleted = self.x_tilde_average
            self.new_adj_without_norm = self.new_adj_output

            self.new_adj_output = self.normalize_graph(self.new_adj_output)   # this time normalize the graph with D-1/2A D-1/2

        ####!!!!!!!the self.new_adj_output is the new adj we got from generator  it is f(X) for the reg loss
        # _,self.ori_logits,_ = self.d_GCN(self.inputs, self.new_adj_output)
        # _,self.mod_pred,_ = self.d_GCN(self.inputs, self.adj_dense, reuse = True)

        return


    def delete_k_edge_old(self,new_adj,ori_adj,  k = 1):
        zeros = tf.zeros_like(new_adj)
        bool_adj = tf.cast(ori_adj, tf.bool)
        new_adj_for_del = tf.where(bool_adj, x =new_adj ,y =zeros ,name = "delete_mask")
        new_adj_for_del = tf.reshape(new_adj_for_del, [-1])
        new_adj = tf.reshape(new_adj, [-1])
        new_indexes = tf.nn.top_k(new_adj_for_del, k = k)
        #new_pos = tf.argmax(new_adj)
        self.mask = tf.ones_like(new_adj, dtype = tf.float32)
        self.mask_onehot = tf.one_hot(new_indexes[1], new_adj.shape[0], dtype=tf.float32)
        self.mask = self.mask - self.mask_onehot
        #self.update_mask= tf.assign(self.mask[new_pos], 0)
        new_adj = tf.multiply(new_adj, self.mask)
        new_adj= tf.reshape(new_adj, [self.n_samples ,self.n_samples])
        return new_adj

    def delete_k_edge_min(self, new_adj, ori_adj, k=3):
        ones = tf.ones_like(new_adj)
        max_value = tf.reduce_max(new_adj)
        lower_bool_label = tf.linalg.band_part(ori_adj,-1,0)
        upper_ori_label = ori_adj - lower_bool_label   # there is no diagnal
        upper_bool_label = tf.cast(upper_ori_label, tf.bool)
        ##############
        self.upper_bool_label = upper_bool_label
        ##############
        new_adj_for_del = tf.where(upper_bool_label, x=new_adj, y=ones * max_value, name="delete_mask")
        self.new_adj_for_del_test = max_value - new_adj_for_del
        new_adj_for_del = max_value - new_adj_for_del
        ori_adj_diag = tf.matrix_diag(tf.matrix_diag_part(ori_adj))  # diagnal matrix
        new_adj_diag = tf.matrix_diag(tf.matrix_diag_part(new_adj))  # diagnal matrix
        ori_adj_diag = tf.reshape(ori_adj_diag, [-1])
        new_adj_diag = tf.reshape(new_adj_diag, [-1])
        new_adj_flat = tf.reshape(new_adj, [-1])
        ori_adj_flat = tf.reshape(ori_adj, [-1])
        # doing the softmax function
        new_adj_for_del_exp = tf.exp(new_adj_for_del)
        new_adj_for_del_exp = tf.where(upper_bool_label, x=new_adj_for_del_exp, y=tf.zeros_like(new_adj_for_del_exp), name="softmax_mask")
        new_adj_for_del_softmax = new_adj_for_del_exp / tf.reduce_sum(new_adj_for_del_exp)
        new_adj_for_del_softmax = tf.reshape(new_adj_for_del_softmax, [-1])
        self.new_adj_for_del_softmax  = new_adj_for_del_softmax
        new_indexes = tf.multinomial(tf.log([new_adj_for_del_softmax]), FLAGS.k)
        ######################## debug
        self.new_indexes = new_indexes
        ########################
        self.mask = upper_ori_label
        self.mask = tf.reshape(self.mask, [-1])
        for i in range(k):
            self.mask_onehot = tf.one_hot(new_indexes[0][i], new_adj_flat.shape[0], dtype=tf.float32)
            self.mask = self.mask - self.mask_onehot
        ######################################  debug
        #self.mask_test = tf.one_hot(new_indexes[1][2], new_adj_flat.shape[0], dtype=tf.float32)
        ######################################
        # self.update_mask= tf.assign(self.mask[new_pos], 0)
        new_adj_out = tf.multiply(new_adj_flat, self.mask)   # the upper triangular
        ori_adj_out = tf.multiply(ori_adj_flat, self.mask)
        # add the transpose and the lower part of the model
        #new_adj_out = new_adj_out + new_adj_diag
        ori_adj_out = ori_adj_out + ori_adj_diag
        ## having the softmax
        #new_adj_out = tf.nn.softmax(new_adj_out)
        #ori_adj_out = tf.nn.softmax(ori_adj_out)
        #new_adj_out = tf.reshape(new_adj_out,[self.n_samples, self.n_samples] )
        ori_adj_out = tf.reshape(ori_adj_out, [self.n_samples, self.n_samples])
        # make the matrix system
        #new_adj_out = new_adj_out + (tf.transpose(new_adj_out) - tf.matrix_diag(tf.matrix_diag_part(new_adj_out)))
        ori_adj_out = ori_adj_out + (tf.transpose(ori_adj_out) - tf.matrix_diag(tf.matrix_diag_part(ori_adj_out)))
        return new_adj_out, ori_adj_out

    def delete_k_edge_min_new(self, new_adj, ori_adj, k=3):  ## this is the newest delete part
        """
        delete the k edges in the new matrix
        :param new_adj:   the x_tilde after the generator
        :param ori_adj:   the original dense adj
        :param k:   how many edges to delete
        :return:  new_adj_out:the deleted edges for x_tilde; ori_adj_out: the deleted edges for original adj
        """
        ones = tf.ones_like(new_adj, dtype = tf.float32)
        max_value = tf.reduce_max(new_adj)
        lower_bool_label = tf.linalg.band_part(ori_adj,-1,0)
        upper_ori_label = ori_adj - lower_bool_label   # there is no diagnal
        upper_bool_label = tf.cast(upper_ori_label, tf.bool)
        ##############
        self.upper_bool_label = upper_bool_label
        ##############
        new_adj_for_del = tf.where(upper_bool_label, x=new_adj, y=ones * max_value, name="delete_mask")
        self.new_adj_for_del_test = max_value - new_adj_for_del
        new_adj_for_del = max_value - new_adj_for_del
        ori_adj_diag = tf.matrix_diag(tf.matrix_diag_part(ori_adj))  # diagnal matrix
        new_adj_diag = tf.matrix_diag(tf.matrix_diag_part(new_adj))  # diagnal matrix
        ori_adj_diag = tf.reshape(ori_adj_diag, [-1])
        #new_adj_diag = tf.reshape(new_adj_diag, [-1])
        new_adj_flat = tf.reshape(new_adj, [-1])
        ori_adj_flat = tf.reshape(ori_adj, [-1])
        # doing the softmax function
        new_adj_for_del_exp = tf.exp(new_adj_for_del)
        new_adj_for_del_exp = tf.where(upper_bool_label, x=new_adj_for_del_exp, y=tf.zeros_like(new_adj_for_del_exp), name="softmax_mask")
        new_adj_for_del_softmax = new_adj_for_del_exp / tf.reduce_sum(new_adj_for_del_exp)
        new_adj_for_del_softmax = tf.reshape(new_adj_for_del_softmax, [-1])
        self.new_adj_for_del_softmax  = new_adj_for_del_softmax
        new_indexes = tf.multinomial(tf.log([new_adj_for_del_softmax]), FLAGS.k)  # this is the sample section
        ######################## debug
        self.new_indexes = new_indexes
        ########################
        self.mask = upper_bool_label
        #self.mask = tf.reshape(self.mask, [-1])
        for i in range(k):
            self.delete_mask_idx = -1 * tf.ones(self.n_samples, dtype=tf.int32)
            self.delete_maskidx_onehot = tf.one_hot(new_indexes[0][i] // self.n_samples, self.n_samples, dtype=tf.int32)
            col_idx = (1 + new_indexes[0][i] % self.n_samples)
            col_idx = tf.cast(col_idx, tf.int32)
            self.delete_mask_idx = self.delete_mask_idx + col_idx * self.delete_maskidx_onehot
            self.delete_onehot_mask = tf.one_hot(self.delete_mask_idx, depth = self.n_samples, dtype = tf.int32)
            self.delete_onehot_mask = tf.cast(self.delete_onehot_mask, tf.bool)
            self.mask = tf.where(self.delete_onehot_mask, x=tf.zeros_like(self.delete_onehot_mask), y=self.mask, name="softmax_mask")
            #self.mask = self.mask - self.delete_onehot_mask
        ######################################  debug
        ######################################
        self.mask = tf.reshape(self.mask, [-1])
        # self.update_mask= tf.assign(self.mask[new_pos], 0)
        # new_adj_out = tf.multiply(new_adj_flat, self.mask)   # the upper triangular
        # ori_adj_out = tf.multiply(ori_adj_flat, self.mask)
        new_adj_out = tf.where(self.mask, x = new_adj_flat, y = tf.zeros_like(new_adj_flat), name = "mask_new_adj")
        ori_adj_out = tf.where(self.mask ,x = ori_adj_flat, y = tf.zeros_like(ori_adj_flat), name = "mask_ori_adj")
        # add the transpose and the lower part of the model
        #new_adj_out = new_adj_out + new_adj_diag
        ori_adj_out = ori_adj_out + ori_adj_diag
        ## having the softmax
        ori_adj_out = tf.reshape(ori_adj_out, [self.n_samples, self.n_samples])
        # make the matrix system
        #new_adj_out = new_adj_out + (tf.transpose(new_adj_out) - tf.matrix_diag(tf.matrix_diag_part(new_adj_out)))
        ori_adj_out = ori_adj_out + (tf.transpose(ori_adj_out) - tf.matrix_diag(tf.matrix_diag_part(ori_adj_out)))
        self.ori_adj_out = ori_adj_out
        return new_adj_out, ori_adj_out

    def delete_k_edge_max_new(self, new_adj, ori_adj, k=3):
        zeros = tf.zeros_like(new_adj)
        #max_value = tf.reduce_max(new_adj)
        lower_bool_label = tf.linalg.band_part(ori_adj,-1,0)
        upper_ori_label = ori_adj - lower_bool_label   # there is no diagnal
        upper_bool_label = tf.cast(upper_ori_label, tf.bool)
        new_adj_for_del = tf.where(upper_bool_label, x=new_adj, y=zeros , name="delete_mask")
        #new_adj_for_del = tf.reshape(new_adj_for_del, [-1])
        #new_adj_for_del = max_value - new_adj_for_del
        ori_adj_diag = tf.matrix_diag(tf.matrix_diag_part(ori_adj))  # diagnal matrix
        new_adj_diag = tf.matrix_diag(tf.matrix_diag_part(new_adj))  # diagnal matrix
        ori_adj_diag = tf.reshape(ori_adj_diag, [-1])
        new_adj_diag = tf.reshape(new_adj_diag, [-1])
        new_adj_flat = tf.reshape(new_adj, [-1])
        ori_adj_flat = tf.reshape(ori_adj, [-1])
        #new_adj_for_del = tf.nn.softmax(new_adj_for_del)
        # doing the softmax function
        new_adj_for_del_exp = tf.exp(new_adj_for_del)
        new_adj_for_del_exp = tf.where(upper_bool_label, x=new_adj_for_del_exp, y=tf.zeros_like(new_adj_for_del_exp),
                                       name="softmax_mask")
        new_adj_for_del_softmax = new_adj_for_del_exp / tf.reduce_sum(new_adj_for_del_exp)
        new_adj_for_del_softmax = tf.reshape(new_adj_for_del_softmax, [-1])
        self.new_adj_for_del_softmax = new_adj_for_del_softmax
        #new_indexes = tf.nn.top_k(new_adj_for_del, k=k)
        new_indexes = tf.multinomial(tf.log([new_adj_for_del_softmax]), FLAGS.k)
        ######################## debug
        self.new_indexes = new_indexes
        ########################
        self.mask = upper_ori_label
        for i in range(k):
            self.delete_mask_idx = -1 * tf.ones(self.n_samples, dtype=tf.int64)
            self.delete_maskidx_onehot = tf.one_hot(new_indexes[0][i] // self.n_samples, self.n_samples, dtype=tf.int64)
            self.delete_mask_idx = self.delete_mask_idx + (
                        1 + new_indexes[0][i] % self.n_samples) * self.delete_maskidx_onehot
            self.delete_onehot_mask = tf.one_hot(self.delete_mask_idx, depth=self.n_samples)
            self.mask = self.mask - self.delete_onehot_mask
        self.mask = tf.reshape(self.mask, [-1])
        ######################################  debug
        #self.mask_test = tf.one_hot(new_indexes[1][2], new_adj_flat.shape[0], dtype=tf.float32)
        ######################################
        # self.update_mask= tf.assign(self.mask[new_pos], 0)
        new_adj_out = tf.multiply(new_adj_flat, self.mask)   # the upper triangular
        ori_adj_out = tf.multiply(ori_adj_flat, self.mask)
        # add the transpose and the lower part of the model
        #new_adj_out = new_adj_out + new_adj_diag
        ori_adj_out = ori_adj_out + ori_adj_diag
        ## having the softmax
        # new_adj_out = tf.nn.softmax(new_adj_out)
        # ori_adj_out = tf.nn.softmax(ori_adj_out)
        #new_adj_out = tf.reshape(new_adj_out,[self.n_samples, self.n_samples] )
        ori_adj_out = tf.reshape(ori_adj_out, [self.n_samples, self.n_samples])
        # make the matrix system
        #new_adj_out = new_adj_out + (tf.transpose(new_adj_out) - tf.matrix_diag(tf.matrix_diag_part(new_adj_out)))
        ori_adj_out = ori_adj_out + (tf.transpose(ori_adj_out) - tf.matrix_diag(tf.matrix_diag_part(ori_adj_out)))
        return new_adj_out, ori_adj_out

    def delete_k_edge_max(self, new_adj, ori_adj, k=3):
        zeros = tf.zeros_like(new_adj)
        #max_value = tf.reduce_max(new_adj)
        lower_bool_label = tf.linalg.band_part(ori_adj,-1,0)
        upper_ori_label = ori_adj - lower_bool_label   # there is no diagnal
        upper_bool_label = tf.cast(upper_ori_label, tf.bool)
        new_adj_for_del = tf.where(upper_bool_label, x=new_adj, y=zeros , name="delete_mask")
        #new_adj_for_del = tf.reshape(new_adj_for_del, [-1])
        #new_adj_for_del = max_value - new_adj_for_del
        ori_adj_diag = tf.matrix_diag(tf.matrix_diag_part(ori_adj))  # diagnal matrix
        new_adj_diag = tf.matrix_diag(tf.matrix_diag_part(new_adj))  # diagnal matrix
        ori_adj_diag = tf.reshape(ori_adj_diag, [-1])
        new_adj_diag = tf.reshape(new_adj_diag, [-1])
        new_adj_flat = tf.reshape(new_adj, [-1])
        ori_adj_flat = tf.reshape(ori_adj, [-1])
        #new_adj_for_del = tf.nn.softmax(new_adj_for_del)
        # doing the softmax function
        new_adj_for_del_exp = tf.exp(new_adj_for_del)
        new_adj_for_del_exp = tf.where(upper_bool_label, x=new_adj_for_del_exp, y=tf.zeros_like(new_adj_for_del_exp),
                                       name="softmax_mask")
        new_adj_for_del_softmax = new_adj_for_del_exp / tf.reduce_sum(new_adj_for_del_exp)
        new_adj_for_del_softmax = tf.reshape(new_adj_for_del_softmax, [-1])
        self.new_adj_for_del_softmax = new_adj_for_del_softmax
        #new_indexes = tf.nn.top_k(new_adj_for_del, k=k)
        new_indexes = tf.multinomial(tf.log([new_adj_for_del_softmax]), FLAGS.k)
        ######################## debug
        self.new_indexes = new_indexes
        ########################
        self.mask = upper_ori_label
        self.mask = tf.reshape(self.mask, [-1])
        for i in range(k):
            self.mask_onehot = tf.one_hot(new_indexes[0][i], new_adj_flat.shape[0], dtype=tf.float32)
            self.mask = self.mask - self.mask_onehot
        ######################################  debug
        #self.mask_test = tf.one_hot(new_indexes[1][2], new_adj_flat.shape[0], dtype=tf.float32)
        ######################################
        # self.update_mask= tf.assign(self.mask[new_pos], 0)
        new_adj_out = tf.multiply(new_adj_flat, self.mask)   # the upper triangular
        ori_adj_out = tf.multiply(ori_adj_flat, self.mask)
        # add the transpose and the lower part of the model
        #new_adj_out = new_adj_out + new_adj_diag
        ori_adj_out = ori_adj_out + ori_adj_diag
        ## having the softmax
        # new_adj_out = tf.nn.softmax(new_adj_out)
        # ori_adj_out = tf.nn.softmax(ori_adj_out)
        #new_adj_out = tf.reshape(new_adj_out,[self.n_samples, self.n_samples] )
        ori_adj_out = tf.reshape(ori_adj_out, [self.n_samples, self.n_samples])
        # make the matrix system
        #new_adj_out = new_adj_out + (tf.transpose(new_adj_out) - tf.matrix_diag(tf.matrix_diag_part(new_adj_out)))
        ori_adj_out = ori_adj_out + (tf.transpose(ori_adj_out) - tf.matrix_diag(tf.matrix_diag_part(ori_adj_out)))
        return new_adj_out, ori_adj_out
    def delete_k_edge(self,new_adj,ori_adj,  k = 3):
        zeros = tf.zeros_like(new_adj)
        bool_adj = tf.cast(ori_adj, tf.bool)
        new_adj_for_del = tf.where(bool_adj, x =new_adj ,y =zeros ,name = "delete_mask")
        new_adj_for_del = tf.reshape(new_adj_for_del, [-1])
        new_adj_flat = tf.reshape(new_adj, [-1])
        ori_adj_flat = tf.reshape(ori_adj, [-1])

        new_indexes = tf.nn.top_k(new_adj_for_del, k = k)
        self.new_indexes = tf.nn.top_k(new_adj_for_del, k=k)
        self.mask = tf.reshape(ori_adj, [-1])
        for i in range(k):
            self.mask_onehot = tf.one_hot(new_indexes[1][i], new_adj_flat.shape[0], dtype=tf.float32)
            self.mask = self.mask - self.mask_onehot
        self.mask_test = tf.one_hot(new_indexes[1][2], new_adj_flat.shape[0], dtype=tf.float32)
        #self.update_mask= tf.assign(self.mask[new_pos], 0)
        new_adj_out = tf.multiply(new_adj_flat, self.mask)
        ori_adj_out = tf.multiply(ori_adj_flat, self.mask)
        new_adj_out= tf.reshape(new_adj_out, [self.n_samples ,self.n_samples])
        ori_adj_out = tf.reshape(ori_adj_out, [self.n_samples ,self.n_samples])
        return new_adj_out, ori_adj_out

    def encoder(self, inputs):
        with tf.variable_scope('encoder') as scope:
            self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                                  output_dim=FLAGS.hidden1,
                                                  adj=self.adj,
                                                  features_nonzero=self.features_nonzero,
                                                  act=tf.nn.relu,
                                                  dropout=self.dropout,
                                                  logging=self.logging, name = "encoder_conv1")(inputs)

            self.z_mean = GraphConvolution(input_dim=FLAGS.hidden1,
                                           output_dim=FLAGS.latent_dim,
                                           adj=self.adj,
                                           act=lambda x: x,
                                           dropout=self.dropout,
                                           logging=self.logging,name = "encoder_conv2")(self.hidden1)

            self.z_log_std = GraphConvolution(input_dim=FLAGS.hidden1,
                                              output_dim=FLAGS.latent_dim,
                                              adj=self.adj,
                                              act=lambda x: x,
                                              dropout=self.dropout,
                                              logging=self.logging,name = "encoder_conv3")(self.hidden1)

            z = self.z_mean + tf.random_normal([self.n_samples, FLAGS.latent_dim]) * tf.exp(
                self.z_log_std)  # middle hidden layer
        return z

    def generate_attention(self, input_z, input_dim, graph_dim, reuse = False):
        input_dim = int(input_dim)
        with tf.variable_scope('generate') as scope:
            if reuse == True:
                scope.reuse_variables()
            self.dense1 = Dense(input_dim=FLAGS.latent_dim, output_dim=2 * FLAGS.latent_dim,act = tf.nn.tanh, bias=False, name = "gene_dense_1")
            self.dense2 = Dense(input_dim=2 * FLAGS.latent_dim, output_dim=1,act = tf.nn.sigmoid, bias=False,name = "gene_dense_2")
            self.dense3 = Dense(input_dim=FLAGS.latent_dim, output_dim=1,act = tf.nn.sigmoid, bias=False, name = "gene_dense_3")
            final_update = input_z[0, :] * input_z
            ## the element wise product to replace the current inner product with size n^2*d
            for i in range(1, self.n_samples):
                update_temp = input_z[i, :] * input_z
                final_update = tf.concat([final_update, update_temp], axis=0)
            #final_update_d1 = tf.tanh(self.dense1(final_update))
            final_update_d1 = self.dense1(final_update)
            reconstructions_weights = tf.nn.softmax(self.dense2(final_update_d1))
            reconstructions = reconstructions_weights * final_update
            #reconstructions =tf.sigmoid(self.dense3(reconstructions))
            reconstructions =tf.nn.softmax(self.dense3(reconstructions))
            reconstructions = tf.reshape(reconstructions, [self.n_samples, self.n_samples])
        return reconstructions

    def generate_dense(self, input_z, input_dim, graph_dim, reuse = False):
        input_dim = int(input_dim)
        with tf.variable_scope('generate') as scope:
            if reuse == True:
                scope.reuse_variables()
            # self.dense1 = Dense(input_dim=FLAGS.latent_dim, output_dim=2 * FLAGS.latent_dim,act = tf.nn.tanh, bias=False, name = "gene_dense_1")
            #self.dense2 = Dense(input_dim=2 * FLAGS.latent_dim, output_dim=1, bias=False,act = tf.nn.sigmoid, name = "gene_dense_2")
            #self.dense1 = Dense(input_dim=FLAGS.latent_dim, output_dim=2 * FLAGS.latent_dim, bias=False, name = "gene_dense_1")
            #self.dense2 = Dense(input_dim=2 * FLAGS.latent_dim, output_dim=1, bias=False, name = "gene_dense_2")
            #self.dense3 = Dense(input_dim=FLAGS.hidden2, output_dim=1, bias=False)
            #final_update = input_z[0, :] * input_z
            update_temp = []
            ## the element wise product to replace the current inner product with size n^2*d
            for i in range(0, self.n_samples):
                update_temp.append(input_z[i, :] * input_z)
            final_update = tf.stack(update_temp, axis=0)
            #final_update_d1 = tf.tanh(self.dense1(final_update))
            #final_update_d1 = self.dense1(final_update)
            #final_update_d1 = tf.layers.dense(final_update,2 * FLAGS.latent_dim, use_bias=False, activation = tf.nn.relu, name="gen_dense1" )
            #######################################################
            #self.final_update_d1_test = final_update_d1
            #######################################################
            #reconstructions = tf.nn.softmax(self.dense2(final_update_d1))  # here should be some
            reconstructions = tf.layers.dense(final_update, 1,use_bias=False, activation = tf.nn.sigmoid, name="gen_dense2")
            reconstructions = tf.squeeze(reconstructions)
            #reconstructions = tf.reshape(reconstructions, [self.n_samples, self.n_samples])
        return reconstructions



    def discriminate_mock_detect(self, inputs,new_adj,  reuse = False):
        # this methods uses this part to mock the community detection algorithm
        with tf.variable_scope('discriminate') as scope:
            if reuse == True:
                scope.reuse_variables()

            self.dis_hidden = GraphConvolutionSparse_denseadj(input_dim=self.input_dim,
                                                  output_dim=FLAGS.hidden1,
                                                  adj=new_adj,
                                                  features_nonzero=self.features_nonzero,
                                                  act=tf.nn.relu,
                                                  dropout=self.dropout,
                                                  logging=self.logging, name ="dis_conv1_sparse")((inputs, new_adj))


            self.dis_z_mean = GraphConvolution_denseadj(input_dim=FLAGS.hidden1,
                                           output_dim=FLAGS.hidden2,
                                           adj=new_adj,
                                           act=lambda x: x,
                                           dropout=self.dropout,
                                           logging=self.logging, name='dis_conv2')((self.dis_hidden, new_adj))
            ############################
            self.dis_z_mean_norm = tf.nn.softmax(self.dis_z_mean, axis = -1)
            for targets in self.target_list:
                targets_indices = [[x] for x in targets]
                #self.G_target_pred = model.vaeD_tilde[targets, :]
                self.Dis_target_pred = tf.gather_nd(self.dis_z_mean_norm, targets_indices)
                self.Dis_target_pred_out = self.Dis_target_pred
                ## calculate the KL divergence
                for i in range(len(targets)):
                    for j in range(i + 1, len(targets)):
                        if ((i == 0) and (j == 1)):
                            self.Dis_comm_loss_KL = tf.reduce_sum(
                                (self.Dis_target_pred[i] * tf.log(self.Dis_target_pred[i] / self.Dis_target_pred[j])))
                        else:
                            self.Dis_comm_loss_KL += tf.reduce_sum((self.Dis_target_pred[i] * tf.log(self.Dis_target_pred[i] / self.Dis_target_pred[j])))
            # to maximize the KL is to minimize the neg KL

            ############################
            self.dis_fully1 =tf.nn.relu(batch_normal(FullyConnect(output_size=256, scope='dis_fully1')(self.dis_z_mean),scope='dis_bn1', reuse = reuse))
            self.dis_output = FullyConnect(output_size = FLAGS.n_clusters, scope='dis_fully2')(self.dis_fully1)
            # the softmax layer for the model
            self.dis_output_softmax = tf.nn.softmax(self.dis_output, axis=-1)
            #self.dis_output = self.dis_z_mean
        return self.dis_output_softmax, self.Dis_comm_loss_KL,self.Dis_target_pred_out

    def d_GCN(self, inputs,new_adj,  reuse = False):
        # this methods uses this part to mock the community detection algorithm
        with tf.variable_scope('discriminate') as scope:
            if reuse == True:
                scope.reuse_variables()

            self.dis_hidden = GraphConvolutionSparse_denseadj(input_dim=self.input_dim,
                                                  output_dim=FLAGS.gcn_hidden1,
                                                  adj=new_adj,
                                                  features_nonzero=self.features_nonzero,
                                                  act=tf.nn.relu,
                                                  dropout=self.dropout,
                                                  logging=self.logging, name ="dis_conv1_sparse")((inputs, new_adj))


            self.dis_z_mean = GraphConvolution_denseadj(input_dim=FLAGS.gcn_hidden1,
                                           output_dim=FLAGS.gcn_hidden2,
                                           adj=new_adj,
                                           act=lambda x: x,
                                           dropout=self.dropout,
                                           logging=self.logging, name='dis_conv2')((self.dis_hidden, new_adj))
            ############################
            self.dis_z_mean_softmax = tf.nn.softmax(self.dis_z_mean, axis = -1)
            self.logits_gather = tf.gather(self.dis_z_mean, self.node_ids)


        return self.dis_z_mean_softmax, self.logits_gather, self.dis_z_mean


    def normalize_graph(self, adj):
        # convert adj into 0 and 1
        #adj_shift = (adj + tf.abs(tf.reduce_min(adj))) / (tf.reduce_max(adj) - tf.reduce_min(adj))
        # self.adj_shift = (adj + tf.abs(tf.reduce_min(adj))) / (tf.reduce_max(adj) - tf.reduce_min(adj))
        # adj = sp.coo_matrix(adj)
        # adj_ = adj + sp.eye(adj.shape[0])
        #adj_ = adj_shift + tf.matrix_diag(tf.ones([self.n_samples], dtype=tf.float32))
        adj_ = adj
        rowsum = tf.reduce_sum(adj_, axis=0)
        # rowsum = np.array(adj_.sum(1))
        degree_mat_inv_sqrt = tf.matrix_diag(tf.pow(rowsum, tf.constant(-0.5)))
        # adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        adj_normalized = tf.matmul(tf.matmul(adj_, tf.transpose(degree_mat_inv_sqrt)), degree_mat_inv_sqrt)
        return adj_normalized


    def normalize_graph_new(self,adj):
        # convert adj into 0 and 1
        rowsum = tf.reduce_sum(adj, axis = -1)
        #rowsum = np.array(adj_.sum(1))
        degree_mat_inv_sqrt = tf.matrix_diag(tf.pow(rowsum, tf.constant(-0.5)))
        adj_normalized = tf.matmul(tf.transpose(tf.matmul(adj, degree_mat_inv_sqrt)), degree_mat_inv_sqrt)
        # adj = sp.coo_matrix(adj)
        # adj_ = adj + sp.eye(adj.shape[0])
        # rowsum = np.array(adj_.sum(1))
        # degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        # adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        return adj_normalized
    pass