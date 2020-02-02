import tensorflow as tf
import numpy as np
flags = tf.app.flags
FLAGS = flags.FLAGS


class OptimizerAE(object):
    def __init__(self, preds, labels, pos_weight, norm):
        preds_sub = preds
        labels_sub = labels

        self.cost = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
                                           tf.cast(labels_sub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


class OptimizerVAE(object):
    def __init__(self, preds, labels, model, num_nodes, pos_weight, norm):
        preds_sub = preds
        labels_sub = labels

        self.cost = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

        # Latent loss
        self.log_lik = self.cost
        self.kl = (0.5 / num_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * model.z_log_std - tf.square(model.z_mean) -
                                                                   tf.square(tf.exp(model.z_log_std)), 1))
        self.cost -= self.kl

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
                                           tf.cast(labels_sub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

class Optimizergaegan(object):
    def __init__(self, preds, labels, model, num_nodes, pos_weight, norm,
                 global_step, new_learning_rate,ori_reg_log,
                 if_drop_edge = True, **kwargs):
        allowed_kwargs = {'placeholders'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        #noised_indexes = kwargs.get("noised_indexes")
        placeholders = kwargs.get("placeholders")
        noised_indexes = placeholders["noised_mask"]
        clean_indexes = placeholders["clean_mask"]
        #noised_indexes = tf.Variable([1,1], dtype = tf.int32,trainable = False)
        en_preds_sub = preds
        en_labels_sub = labels
        self.opt_op = 0  # this is the minimize function
        self.cost = 0  # this is the loss
        self.accuracy = 0  # this is the accuracy
        self.G_comm_loss = 0
        self.G_comm_loss_KL = 0
        self.D_loss = 0
        self.num_nodes = num_nodes
        self.if_drop_edge = if_drop_edge
        self.last_reg = tf.Variable(0,name = "last_reg", dtype = tf.float32, trainable=False)
        self.ori_reg_log = ori_reg_log
        # this is for vae, it contains two parts of losses:
        # self.encoder_optimizer = tf.train.RMSPropOptimizer(learning_rate = new_learning_rate)
        self.generate_optimizer = tf.train.RMSPropOptimizer(learning_rate= new_learning_rate)
        self.discriminate_optimizer = tf.train.RMSPropOptimizer(learning_rate = new_learning_rate)
        # encoder_varlist = [var for var in tf.trainable_variables() if 'encoder' in var.name]
        generate_varlist = [var for var in tf.trainable_variables() if (
                    'generate' in var.name) or ('encoder' in var.name)]  # the first part is generator and the second part is discriminator
        discriminate_varlist = [var for var in tf.trainable_variables() if 'discriminate' in var.name]
        if if_drop_edge == True:
            ############### the loss functions we use to train the model
            #self.G_comm_loss = self.reg_loss_many_samples(model, self.G_comm_loss)
            #self.G_comm_loss = self.reg_loss_many_samples_reward_per(model, self.G_comm_loss)
            #self.G_comm_loss = self.reg_loss_many_samples_reward_ratio_no_reverse(model, self.G_comm_loss)
            #self.G_comm_loss = self.reg_loss_many_samples_reward_ratio_no_reverse_softmax(model, self.G_comm_loss)
            #self.G_comm_loss = self.reg_loss_many_samples_no_reverse_softmax_features(model, self.G_comm_loss)
            # self.G_comm_loss = self.reg_loss_no_smaple_reverse_features_only(model,
            #                                                             self.G_comm_loss)
            #self.G_comm_loss = self.reg_loss_no_sample_reverse_edges_only(model,
            #                                                            self.G_comm_loss)
            # self.G_comm_loss = self.reg_loss_no_sample_reverse_edges_only_ori_current(model,
            #                                                                           self.G_comm_loss)
            #self.G_comm_loss = self.reg_loss_no_sample_reverse_edges_only_ori_current_intersect(model,
            #                                                                          self.G_comm_loss)
            self.G_comm_loss = self.loss_cross_entropy_logits(model, noised_indexes,clean_indexes, self.G_comm_loss)
        ######################################################
        # because the generate part is only inner product , there is no variable to optimize, we should change the format and try again
            if FLAGS.generator == "graphite":
                self.G_min_op = self.generate_optimizer.minimize(self.G_comm_loss, global_step=global_step,
                                                                 var_list=generate_varlist)
            if FLAGS.generator == "inner_product":
                self.G_min_op = self.generate_optimizer.minimize(self.G_comm_loss, global_step=global_step,
                                                                 var_list=generate_varlist)
            if FLAGS.generator == "graphite_noinner":
                self.G_min_op = self.generate_optimizer.minimize(self.G_comm_loss, global_step=global_step,
                                                                 var_list=generate_varlist)
            else:
                self.G_min_op = self.generate_optimizer.minimize(self.G_comm_loss, global_step=global_step,
                                                                 var_list=generate_varlist)
            ######################################################
            self.D_loss = self.dis_cutmin_loss_clean(model)
            self.D_min_op = self.discriminate_optimizer.minimize(self.D_loss, global_step = global_step,
                                                                 var_list = discriminate_varlist)

    def reg_loss_many_samples(self, model, G_comm_loss):
        for idx, x_tilde_deleted in enumerate(model.x_tilde_list):
            self.reg = 0
            ## the Laplacian loss
            x_tilde_deleted_mat = tf.reshape(x_tilde_deleted, shape=[self.num_nodes, self.num_nodes])
            rowsum = tf.reduce_sum(model.adj_ori_dense, axis=1)
            rowsum = tf.matrix_diag(rowsum)
            self.g_delta = rowsum - model.adj_ori_dense
            temp = tf.matmul(tf.transpose(x_tilde_deleted_mat), self.g_delta)
            self.reg_mat = tf.matmul(temp, x_tilde_deleted_mat)
            ###### grab non zero part
            # self.reg = tf.gather_nd(self.reg_mat, tf.where(self.reg_mat > 0))   # set the bigger then
            ###### norm version
            # self.reg = tf.square(tf.norm(self.reg_mat))
            ###### trace version
            self.reg = tf.trace(self.reg_mat)
            ####################
            # self.reg = self.reg * 1e-9
            self.reg = self.reg
            #self.reg = tf.log(self.reg)
            # self.reg = tf.reduce_mean(self.reg_mat)
            self.reg = 1 / (self.reg + 1e-10)

            ## self.G_comm_loss
            eij = tf.gather_nd(model.x_tilde_deleted, tf.where(model.x_tilde_deleted > 0))
            eij = tf.reduce_sum(tf.log(eij))
            # self.G_comm_loss = (-1)* self.mu * eij + FLAGS.G_KL_r * self.G_comm_loss_KL
            if idx == 0:
                G_comm_loss =  (-1) * self.reg * eij
            G_comm_loss += (-1) * self.reg * eij
        G_comm_loss = G_comm_loss / len(model.x_tilde_list)
        return G_comm_loss

    def reg_loss_many_samples_reward_ratio(self, model, G_comm_loss):
        """
        The loss with samples on delete x_tilde and add the reward percentage from Q learning
        :param self:
        :param model:
        :param G_comm_loss:
        :return:
        """
        self.reward_list = []
        self.percentage_all = 0
        for idx, x_tilde_deleted in enumerate(model.x_tilde_list):
            self.reg = 0

            ## the Laplacian loss
            x_tilde_deleted_mat = tf.reshape(x_tilde_deleted, shape=[self.num_nodes, self.num_nodes])
            rowsum = tf.reduce_sum(model.adj_ori_dense, axis=1)
            rowsum = tf.matrix_diag(rowsum)
            self.g_delta = rowsum - model.adj_ori_dense
            temp = tf.matmul(tf.transpose(x_tilde_deleted_mat), self.g_delta)
            self.reg_mat = tf.matmul(temp, x_tilde_deleted_mat)
            ###### grab non zero part
            # self.reg = tf.gather_nd(self.reg_mat, tf.where(self.reg_mat > 0))   # set the bigger then
            ###### norm version
            # self.reg = tf.square(tf.norm(self.reg_mat))
            ###### trace version
            self.reg = tf.trace(self.reg_mat)
            self.reg_trace = self.reg
            ####################
            # self.reg = self.reg * 1e-9
            # self.reg = self.reg
            self.reg = tf.log(self.reg + 1)
            #self.reg_log = self.reg
            self.reg_log = self.reg * 0.1
            # self.reg = tf.reduce_mean(self.reg_mat)
            self.reg = 1 / (self.reg + 1e-10)

            ## self.G_comm_loss
            eij = tf.gather_nd(model.x_tilde_deleted, tf.where(model.x_tilde_deleted > 0))
            eij = tf.reduce_sum(tf.log(eij))
            # self.G_comm_loss = (-1)* self.mu * eij + FLAGS.G_KL_r * self.G_comm_loss_KL
            if idx == 0:
                G_comm_loss_mean = self.reg * eij
                self.reward_list.append(self.reg * eij)
                self.percentage_all = model.reward_percent_list[idx]
            else:
                G_comm_loss_mean += self.reg * eij
                self.reward_list.append(self.reg * eij)
                self.percentage_all += model.reward_percent_list[idx]
        G_comm_loss_mean = G_comm_loss_mean / len(model.x_tilde_list)
        for idx, item in enumerate(self.reward_list):
            if idx == 0:
                G_comm_loss = (model.reward_percent_list[idx] / self.percentage_all) * (item - G_comm_loss_mean)
            else:
                G_comm_loss += (model.reward_percent_list[idx] / self.percentage_all) * (item - G_comm_loss_mean)
        G_comm_loss = (-1) * G_comm_loss
        return G_comm_loss

    def reg_loss_many_samples_reward_ratio_no_reverse(self, model, G_comm_loss):
        """
        The loss with samples on delete x_tilde and add the reward percentage from Q learning
        :param self:
        :param model:
        :param G_comm_loss:
        :return:
        """
        self.reward_list = []
        self.percentage_all = 0
        for idx, x_tilde_deleted in enumerate(model.x_tilde_list):
            self.reg = 0

            ## the Laplacian loss
            x_tilde_deleted_mat = tf.reshape(x_tilde_deleted, shape=[self.num_nodes, self.num_nodes])
            rowsum = tf.reduce_sum(model.adj_ori_dense, axis=1)
            rowsum = tf.matrix_diag(rowsum)
            self.g_delta = rowsum - model.adj_ori_dense
            temp = tf.matmul(tf.transpose(x_tilde_deleted_mat), self.g_delta)
            self.reg_mat = tf.matmul(temp, x_tilde_deleted_mat)
            ###### grab non zero part
            # self.reg = tf.gather_nd(self.reg_mat, tf.where(self.reg_mat > 0))   # set the bigger then
            ###### norm version
            # self.reg = tf.square(tf.norm(self.reg_mat))
            ###### trace version
            self.reg = tf.trace(self.reg_mat)
            self.reg_trace = self.reg
            ####################
            # self.reg = self.reg * 1e-9
            # self.reg = self.reg
            self.reg = tf.log(self.reg + 1)
            # self.reg_log = self.reg
            self.reg_log = self.reg
            # self.reg = tf.reduce_mean(self.reg_mat)
            #self.reg = 1 / (self.reg + 1e-10)

            ## self.G_comm_loss
            #eij = tf.gather_nd(model.x_tilde_deleted, tf.where(model.x_tilde_deleted > 0))
            #eij = tf.reduce_sum(tf.log(eij))
            # self.G_comm_loss = (-1)* self.mu * eij + FLAGS.G_KL_r * self.G_comm_loss_KL
            if idx == 0:
                G_comm_loss_mean = self.reg
                self.reward_list.append(self.reg)
                self.percentage_all = model.reward_percent_list[idx]
            else:
                G_comm_loss_mean += self.reg
                self.reward_list.append(self.reg)
                self.percentage_all += model.reward_percent_list[idx]
        G_comm_loss_mean = G_comm_loss_mean / len(model.x_tilde_list)
        #### if we need the softmax function for this part
        #new_percent_softmax = tf.nn.softmax(model.reward_percent_list)
        ########
        for idx, item in enumerate(self.reward_list):
            if idx == 0:
                G_comm_loss = (model.reward_percent_list[idx] / self.percentage_all) * (item - G_comm_loss_mean)
                #G_comm_loss = (new_percent_softmax[idx]) * (item - G_comm_loss_mean)
            else:
                G_comm_loss += (model.reward_percent_list[idx] / self.percentage_all) * (item - G_comm_loss_mean)
                #G_comm_loss += (new_percent_softmax[idx]) * (item - G_comm_loss_mean)
        #G_comm_loss = (-1) * G_comm_loss
        return G_comm_loss

    def reg_loss_many_samples_reward_ratio_no_reverse_softmax(self, model, G_comm_loss):
        """
        The loss with samples on delete x_tilde and add the reward percentage from Q learning
        :param self:
        :param model:
        :param G_comm_loss:
        :return:
        """
        self.reward_list = []
        self.percentage_all = 0
        G_comm_loss_mean = 0
        for idx, adj_deleted in enumerate(model.new_adj_outlist):
            self.reg = 0

            ## the Laplacian loss
            adj_deleted_mat = tf.reshape(adj_deleted, shape=[self.num_nodes, self.num_nodes])
            rowsum = tf.reduce_sum(adj_deleted_mat, axis=1)
            rowsum = tf.matrix_diag(rowsum)
            self.g_delta = rowsum - adj_deleted_mat
            feature_dense = tf.sparse_tensor_to_dense(model.inputs)
            temp = tf.matmul(tf.transpose(feature_dense), self.g_delta)
            self.reg_mat = tf.matmul(temp, feature_dense)
            ###### grab non zero part
            # self.reg = tf.gather_nd(self.reg_mat, tf.where(self.reg_mat > 0))   # set the bigger then
            ###### norm version
            # self.reg = tf.square(tf.norm(self.reg_mat))
            ###### trace version
            self.reg = tf.trace(self.reg_mat)
            self.reg_trace = self.reg
            ####################
            # self.reg = self.reg * 1e-9
            # self.reg = self.reg
            self.reg = tf.log(self.reg + 1)
            # self.reg_log = self.reg
            self.reg_log = self.reg
            # self.reg = tf.reduce_mean(self.reg_mat)
            #self.reg = 1 / (self.reg + 1e-10)

            ## self.G_comm_loss
            #eij = tf.gather_nd(model.x_tilde_deleted, tf.where(model.x_tilde_deleted > 0))
            #eij = tf.reduce_sum(tf.log(eij))
            # self.G_comm_loss = (-1)* self.mu * eij + FLAGS.G_KL_r * self.G_comm_loss_KL
            if idx == 0:
                G_comm_loss_mean = self.reg
                self.reward_list.append(self.reg)
                self.percentage_all = model.reward_percent_list[idx]
            else:
                G_comm_loss_mean += self.reg
                self.reward_list.append(self.reg)
                self.percentage_all += model.reward_percent_list[idx]
        G_comm_loss_mean = G_comm_loss_mean / len(model.new_adj_outlist)
        #### if we need the softmax function for this part
        new_percent_softmax = tf.nn.softmax(model.reward_percent_list)
        self.new_percent_softmax = new_percent_softmax
        ########
        for idx, item in enumerate(self.reward_list):
            if idx == 0:
                #G_comm_loss = (model.reward_percent_list[idx] / self.percentage_all) * (item - G_comm_loss_mean)
                G_comm_loss = (new_percent_softmax[idx]) * (item - G_comm_loss_mean)
            else:
                #G_comm_loss += (model.reward_percent_list[idx] / self.percentage_all) * (item - G_comm_loss_mean)
                G_comm_loss += (new_percent_softmax[idx]) * (item - G_comm_loss_mean)
        #G_comm_loss = (-1) * G_comm_loss
        return G_comm_loss

    def reg_loss_many_samples_no_reverse_softmax_features(self, model, G_comm_loss):
        """
        The loss with samples on delete x_tilde and add the reward percentage from Q learning
        :param self:
        :param model:
        :param G_comm_loss:
        :return:
        """
        G_comm_loss_mean = 0
        self.reward_list = []
        self.percentage_all = 0
        for idx, adj_deleted in enumerate(model.new_adj_outlist):
            self.reg = 0
            ## the Laplacian loss here we should use the real new one
            adj_deleted_mat = tf.reshape(adj_deleted, shape=[self.num_nodes, self.num_nodes])
            rowsum = tf.reduce_sum(adj_deleted_mat, axis=1)
            rowsum = tf.matrix_diag(rowsum)
            self.g_delta = rowsum - adj_deleted_mat
            temp = tf.matmul(tf.transpose(model.new_features_list[idx]), self.g_delta)
            self.reg_mat = tf.matmul(temp, model.new_features_list[idx])
            ###### grab non zero part
            # self.reg = tf.gather_nd(self.reg_mat, tf.where(self.reg_mat > 0))   # set the bigger then
            ###### norm version
            # self.reg = tf.square(tf.norm(self.reg_mat))
            ###### trace version
            self.reg = tf.trace(self.reg_mat)
            self.reg_trace = self.reg
            #self.reg = tf.log(self.reg + 1)
            self.reg = tf.log(self.reg)
            # self.reg_log = self.reg
            self.reg_log = self.reg
            # self.reg = tf.reduce_mean(self.reg_mat)
            # self.reg = 1 / (self.reg + 1e-10)

            ## self.G_comm_loss
            # eij = tf.gather_nd(model.x_tilde_deleted, tf.where(model.x_tilde_deleted > 0))
            # eij = tf.reduce_sum(tf.log(eij))
            # self.G_comm_loss = (-1)* self.mu * eij + FLAGS.G_KL_r * self.G_comm_loss_KL
            if idx == 0:
                G_comm_loss_mean = self.reg
                self.reward_list.append(self.reg)
                self.percentage_all = model.percentage_list_all[idx]
            else:
                G_comm_loss_mean += self.reg
                self.reward_list.append(self.reg)
                self.percentage_all += model.percentage_list_all[idx]
        G_comm_loss_mean = G_comm_loss_mean / len(model.new_adj_outlist)
        #### if we need the softmax function for this part
        new_percent_softmax = tf.nn.softmax(model.percentage_list_all)
        self.new_percent_softmax = new_percent_softmax
        ########
        for idx, item in enumerate(self.reward_list):
            if idx == 0:
                # G_comm_loss = (model.reward_percent_list[idx] / self.percentage_all) * (item - G_comm_loss_mean)
                G_comm_loss = (new_percent_softmax[idx]) * (item - G_comm_loss_mean)
            else:
                # G_comm_loss += (model.reward_percent_list[idx] / self.percentage_all) * (item - G_comm_loss_mean)
                G_comm_loss += (new_percent_softmax[idx]) * (item - G_comm_loss_mean)
        # G_comm_loss = (-1) * G_comm_loss
        return G_comm_loss

    def reg_loss_no_smaple_reverse_features_only(self, model, G_comm_loss):
        """
        The loss with samples on delete x_tilde and add the reward percentage from Q learning
        :param self:
        :param model:
        :param G_comm_loss:
        :return:
        """
        self.reward_list = []
        self.percentage_all = 0
        for idx, adj_deleted in enumerate(model.new_adj_outlist):
            self.reg = 0
            adj_deleted_mat = model.adj_ori_dense
            ### the Laplacian loss here we should use the real new one
            #adj_deleted_mat = tf.reshape(adj_deleted, shape=[self.num_nodes, self.num_nodes])
            rowsum = tf.reduce_sum(adj_deleted_mat, axis=1)
            rowsum = tf.matrix_diag(rowsum)
            self.g_delta = rowsum - adj_deleted_mat
            temp = tf.matmul(tf.transpose(model.new_features_list[idx]), self.g_delta)
            self.reg_mat = tf.matmul(temp, model.new_features_list[idx])
            ###### grab non zero part
            # self.reg = tf.gather_nd(self.reg_mat, tf.where(self.reg_mat > 0))   # set the bigger then
            ###### norm version
            # self.reg = tf.square(tf.norm(self.reg_mat))
            ###### trace version
            self.reg = tf.trace(self.reg_mat)
            self.reg_trace = self.reg
            #self.reg = tf.log(self.reg + 1)
            self.reg = tf.log(self.reg)
            # self.reg_log = self.reg
            self.reg_log = self.reg
            # self.reg = tf.reduce_mean(self.reg_mat)
            self.reg = 1 / (self.reg + 1e-10)

            ## self.G_comm_loss
            # eij = tf.gather_nd(model.x_tilde_deleted, tf.where(model.x_tilde_deleted > 0))
            # eij = tf.reduce_sum(tf.log(eij))
            # self.G_comm_loss = (-1)* self.mu * eij + FLAGS.G_KL_r * self.G_comm_loss_KL
            if idx == 0:
                #G_comm_loss_mean = self.reg
                self.reward_list.append(self.reg)
                self.percentage_all = model.percentage_list_all[idx]
                self.percentage_fea = model.percentage_fea[idx]
            #else:
            #    G_comm_loss_mean += self.reg
            #    self.reward_list.append(self.reg)
            #    self.percentage_all += model.percentage_list_all[idx]
        #G_comm_loss_mean = G_comm_loss_mean / len(model.new_adj_outlist)
        #### if we need the softmax function for this part
        # new_percent_softmax = tf.nn.softmax(model.percentage_list_all)
        # self.new_percent_softmax = new_percent_softmax
        ########
        for idx, item in enumerate(self.reward_list):
            if idx == 0:
                # G_comm_loss = (model.reward_percent_list[idx] / self.percentage_all) * (item - G_comm_loss_mean)
                G_comm_loss = (self.percentage_fea) * (item)
            #else:
            #    # G_comm_loss += (model.reward_percent_list[idx] / self.percentage_all) * (item - G_comm_loss_mean)
            #    G_comm_loss += (new_percent_softmax[idx]) * (item - G_comm_loss_mean)
        G_comm_loss = (-1) * G_comm_loss
        return G_comm_loss

    def reg_loss_no_sample_reverse_edges_only(self, model, G_comm_loss):
        """
        The loss with samples on delete x_tilde and add the reward percentage from Q learning
        :param self:
        :param model:
        :param G_comm_loss:
        :return:
        """
        self.reward_list = []
        self.percentage_all = 0
        for idx, adj_deleted in enumerate(model.new_adj_outlist):
            self.reg = 0
            # adj_deleted_mat = model.adj_ori_dense
            ### the Laplacian loss here we should use the real new one
            adj_deleted_mat = tf.reshape(adj_deleted, shape=[self.num_nodes, self.num_nodes])
            rowsum = tf.reduce_sum(adj_deleted_mat, axis=1)
            rowsum = tf.matrix_diag(rowsum)
            self.g_delta = rowsum - adj_deleted_mat
            temp = tf.matmul(tf.transpose(model.feature_dense), self.g_delta)
            self.reg_mat = tf.matmul(temp, model.feature_dense)
            ###### grab non zero part
            # self.reg = tf.gather_nd(self.reg_mat, tf.where(self.reg_mat > 0))   # set the bigger then
            ###### norm version
            # self.reg = tf.square(tf.norm(self.reg_mat))
            ###### trace version
            self.reg = tf.trace(self.reg_mat)
            self.reg_trace = self.reg
            #self.reg = tf.log(self.reg + 1)
            self.reg = tf.log(self.reg)
            # self.reg_log = self.reg
            self.reg_log = self.reg
            #self.reg = self.reg * 0.1
            #self.reg = 1 / (self.reg + 1e-10)
            #### L1 - L2 to replace the laplacian
            self.reg = self.last_reg - self.reg
            self.reg = FLAGS.reward_para * self.reg
            #####
            ## self.G_comm_loss
            # eij = tf.gather_nd(model.x_tilde_deleted, tf.where(model.x_tilde_deleted > 0))
            # eij = tf.reduce_sum(tf.log(eij))
            # self.G_comm_loss = (-1)* self.mu * eij + FLAGS.G_KL_r * self.G_comm_loss_KL
            self.percentage_edge = model.reward_percent_list[0]
            #else:
            #    G_comm_loss_mean += self.reg
            #    self.reward_list.append(self.reg)
            #    self.percentage_all += model.percentage_list_all[idx]
        #G_comm_loss_mean = G_comm_loss_mean / len(model.new_adj_outlist)
        #### if we need the softmax function for this part
        # new_percent_softmax = tf.nn.softmax(model.percentage_list_all)
        # self.new_percent_softmax = new_percent_softmax
        ########
                # G_comm_loss = (model.reward_percent_list[idx] / self.percentage_all) * (item - G_comm_loss_mean)
            G_comm_loss = (self.percentage_edge) * (self.reg)
            #else:
            #    # G_comm_loss += (model.reward_percent_list[idx] / self.percentage_all) * (item - G_comm_loss_mean)
            #    G_comm_loss += (new_percent_softmax[idx]) * (item - G_comm_loss_mean)
        G_comm_loss = (-1) * G_comm_loss
        return G_comm_loss
    def reg_loss_no_sample_reverse_edges_only_ori_current(self, model,
                                                          G_comm_loss):
        """
        The loss with samples on delete x_tilde and add the reward percentage from Q learning
        :param self:
        :param model:
        :param G_comm_loss:
        :return:
        """
        self.reward_list = []
        self.percentage_all = 0
        for idx, adj_deleted in enumerate(model.new_adj_outlist):
            self.reg = 0
            # adj_deleted_mat = model.adj_ori_dense
            ### the Laplacian loss here we should use the real new one
            adj_deleted_mat = tf.reshape(adj_deleted, shape=[self.num_nodes, self.num_nodes])
            rowsum = tf.reduce_sum(adj_deleted_mat, axis=1)
            rowsum = tf.matrix_diag(rowsum)
            self.g_delta = rowsum - adj_deleted_mat
            temp = tf.matmul(tf.transpose(model.feature_dense), self.g_delta)
            self.reg_mat = tf.matmul(temp, model.feature_dense)
            ###### grab non zero part
            # self.reg = tf.gather_nd(self.reg_mat, tf.where(self.reg_mat > 0))   # set the bigger then
            ###### norm version
            # self.reg = tf.square(tf.norm(self.reg_mat))
            ###### trace version
            self.reg = tf.trace(self.reg_mat)
            self.reg_trace = self.reg
            #self.reg = tf.log(self.reg + 1)
            self.reg = tf.log(self.reg)
            # self.reg_log = self.reg
            self.reg_log = self.reg
            #self.reg = self.reg * 0.1
            #self.reg = 1 / (self.reg + 1e-10)
            #### L1 - L2 to replace the laplacian
            self.reg = self.ori_reg_log - self.reg
            self.reg = FLAGS.reward_para * self.reg
            #####
            ## self.G_comm_loss
            # eij = tf.gather_nd(model.x_tilde_deleted, tf.where(model.x_tilde_deleted > 0))
            # eij = tf.reduce_sum(tf.log(eij))
            # self.G_comm_loss = (-1)* self.mu * eij + FLAGS.G_KL_r * self.G_comm_loss_KL
            self.percentage_edge = model.reward_percent_list[0]
            #else:
            #    G_comm_loss_mean += self.reg
            #    self.reward_list.append(self.reg)
            #    self.percentage_all += model.percentage_list_all[idx]
        #G_comm_loss_mean = G_comm_loss_mean / len(model.new_adj_outlist)
        #### if we need the softmax function for this part
        # new_percent_softmax = tf.nn.softmax(model.percentage_list_all)
        # self.new_percent_softmax = new_percent_softmax
        ########
                # G_comm_loss = (model.reward_percent_list[idx] / self.percentage_all) * (item - G_comm_loss_mean)
            G_comm_loss = (self.percentage_edge) * (self.reg)
            #else:
            #    # G_comm_loss += (model.reward_percent_list[idx] / self.percentage_all) * (item - G_comm_loss_mean)
            #    G_comm_loss += (new_percent_softmax[idx]) * (item - G_comm_loss_mean)
        G_comm_loss = (-1) * G_comm_loss
        return G_comm_loss

    def reg_loss_no_sample_reverse_edges_only_ori_current_density(self, model,
                                                          G_comm_loss):
        """
        The loss with samples on delete x_tilde and add the reward percentage from Q learning
        :param self:
        :param model:
        :param G_comm_loss:
        :return:
        """
        self.reward_list = []
        self.percentage_all = 0
        for idx, adj_deleted in enumerate(model.new_adj_outlist):
            self.reg = 0
            #### L1 - L2 to replace the laplacian
            self.reg = model.vaeD_density      ## bigger is better
            self.reg = FLAGS.reward_para * self.reg
            #####
            self.percentage_edge = model.reward_percent_list[0]
            G_comm_loss = (self.percentage_edge) * (self.reg)
            self.reward_and_per = (self.percentage_edge) * (self.reg)
        G_comm_loss = (-1) * G_comm_loss
        return G_comm_loss
    def reg_loss_no_sample_reverse_edges_only_ori_current_intersect(self, model,
                                                          G_comm_loss):
        """
        The loss with samples on delete x_tilde and add the reward percentage from Q learning
        :param self:
        :param model:
        :param G_comm_loss:
        :return:
        """
        self.reward_list = []
        self.percentage_all = 0
        for idx, adj_deleted in enumerate(model.new_adj_outlist):
            self.reg = 0
            #### L1 - L2 to replace the laplacian
            self.reg = tf.cast(model.inter_num, tf.float32)      ## bigger is better

            self.reg = FLAGS.reward_para * self.reg

            #####
            self.percentage_edge = model.reward_percent_list[0]
            G_comm_loss = (self.percentage_edge) * (self.reg)
            self.reward_and_per = (self.percentage_edge) * (self.reg)
        G_comm_loss = (-1) * G_comm_loss
        return G_comm_loss
    def loss_cross_entropy_logits(self, model,noised_indexes, clean_indexes, G_comm_loss):
        """
        The loss with samples on delete x_tilde and add the reward percentage from Q learning
        :param self:
        :param model:
        :param G_comm_loss:
        :return:
        """
        noised_indexes_2d = tf.stack([noised_indexes //self.num_nodes,
                                      noised_indexes % self.num_nodes], axis = -1)
        clean_indexes_2d = tf.stack([clean_indexes // self.num_nodes,
                                     clean_indexes % self.num_nodes], axis =-1)
        adj_ori = model.adj_ori_dense - \
            tf.matrix_diag(tf.diag_part(model.adj_ori_dense))
        #clean_mask = tf.where(tf.equal(adj_ori, 1))
        clean_mask = clean_indexes_2d
        real_pred = tf.gather_nd(model.x_tilde_output_ori,clean_mask)
        fake_pred = tf.gather_nd(model.x_tilde_output_ori, noised_indexes_2d)
        loss_real = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(real_pred), logits = real_pred)
        loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros_like(fake_pred), logits = fake_pred)
        G_comm_loss = tf.reduce_mean(loss_real) +tf.reduce_mean(loss_fake)
        # self.reward_list = []
        # self.percentage_all = 0
        # for idx, adj_deleted in enumerate(model.new_adj_outlist):
        #    self.reg = 0
        #    #### L1 - L2 to replace the laplacian
        #    self.reg = tf.cast(model.inter_num, tf.float32)      ## bigger is better

        #    self.reg = FLAGS.reward_para * self.reg

            #####
        #    self.percentage_edge = model.reward_percent_list[0]
        #    G_comm_loss = (self.percentage_edge) * (self.reg)
        #    self.reward_and_per = (self.percentage_edge) * (self.reg)
        #G_comm_loss = (-1) * G_comm_loss
        return G_comm_loss



    def dis_cutmin_loss_clean(self, model):
        A_pool = tf.matmul(
            tf.transpose(tf.matmul(model.adj_ori_dense, model.realD_tilde)), model.realD_tilde)
        num = tf.diag_part(A_pool)

        D = tf.reduce_sum(model.adj_ori_dense, axis=-1)
        D = tf.matrix_diag(D)
        D_pooled = tf.matmul(
            tf.transpose(tf.matmul(D, model.realD_tilde)), model.realD_tilde)
        den = tf.diag_part(D_pooled)
        D_mincut_loss = -(1 / FLAGS.n_clusters) * (num / den)
        D_mincut_loss = tf.reduce_sum(D_mincut_loss)
        ## the orthogonal part loss
        St_S = (FLAGS.n_clusters / self.num_nodes) * tf.matmul(tf.transpose(model.realD_tilde), model.realD_tilde)
        I_S = tf.eye(FLAGS.n_clusters)
        # ortho_loss =tf.norm(St_S / tf.norm(St_S) - I_S / tf.norm(I_S))
        ortho_loss = tf.square(tf.norm(St_S - I_S))
        # S_T = tf.transpose(model.vaeD_tilde, perm=[1, 0])
        # AA_T = tf.matmul(model.vaeD_tilde, S_T) - tf.eye(FLAGS.n_clusters)
        # ortho_loss = tf.square(tf.norm(AA_T))
        ## the overall cutmin_loss
        D_loss = D_mincut_loss + FLAGS.mincut_r * ortho_loss
        return D_loss

    pass

