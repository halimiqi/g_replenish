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
    def __init__(self, preds, labels, model, num_nodes, pos_weight, norm, global_step, new_learning_rate, if_drop_edge = True):
        en_preds_sub = preds
        en_labels_sub = labels
        self.opt_op = 0  # this is the minimize function
        self.cost = 0  # this is the loss
        self.accuracy = 0  # this is the accuracy
        self.G_comm_loss = 0
        self.G_comm_loss_KL = 0
        self.num_nodes = num_nodes
        self.if_drop_edge = if_drop_edge
        # this is for vae, it contains two parts of losses:
        # self.encoder_optimizer = tf.train.RMSPropOptimizer(learning_rate = new_learning_rate)
        self.generate_optimizer = tf.train.RMSPropOptimizer(learning_rate= new_learning_rate)
        self.discriminate_optimizer = tf.train.RMSPropOptimizer(learning_rate = new_learning_rate)
        # encoder_varlist = [var for var in tf.trainable_variables() if 'encoder' in var.name]
        generate_varlist = [var for var in tf.trainable_variables() if (
                    'generate' in var.name) or ('encoder' in var.name)]  # the first part is generator and the second part is discriminator
        discriminate_varlist = [var for var in tf.trainable_variables() if 'discriminate' in var.name]
        #################### the new G_comm_loss
        # for targets in target_list:
        #     targets_indices = [[x] for x in targets]
        #     #self.G_target_pred = model.vaeD_tilde[targets, :]
        #     self.G_target_pred = tf.gather_nd(model.vaeD_tilde, targets_indices)
        #     ## calculate the KL divergence
        #     for i in range(len(targets)):
        #         for j in range(i + 1, len(targets)):
        #             if (i == 0) and (j == 1):
        #                 self.G_comm_loss_KL = -1 * tf.reduce_sum(
        #                     (self.G_target_pred[i] * tf.log(self.G_target_pred[i] / self.G_target_pred[j])))
        #             else:
        #                 self.G_comm_loss_KL += -1*tf.reduce_sum((self.G_target_pred[i] * tf.log(self.G_target_pred[i] / self.G_target_pred[j])))
                    ## to maximize the KL is to minimize the neg KL
        ######################################################


        ######################################################
        # if if_drop_edge == True:
        #     self.mu = 0
        #     ## the new G_comm_loss
        #     for idx, targets in enumerate(target_list):
        #         target_pred = tf.gather(model.vaeD_tilde, targets)
        #         max_index = tf.argmax(target_pred, axis=1)
        #         max_index = tf.cast(max_index, tf.int32)
        #         if idx == 0:
        #             self.mu = ((len(tf.unique(max_index)) - 1) / (
        #                         np.max([FLAGS.n_clusters - 1, 1]) * (tf.reduce_max(tf.bincount(max_index)))))
        #         else:
        #             self.mu += ((len(tf.unique(max_index)) - 1) / (
        #                         np.max([FLAGS.n_clusters - 1, 1]) * (tf.reduce_max(tf.bincount(max_index)))))
        #     self.mu = tf.cast(self.mu, tf.float32)
        #     eij = tf.gather_nd(model.x_tilde_deleted, tf.where(model.x_tilde_deleted > 0))
        #     eij = tf.reduce_sum(tf.log(eij))
        #     #self.G_comm_loss = (-1)* self.mu * eij + FLAGS.G_KL_r * self.G_comm_loss_KL
        #     self.G_comm_loss = (-1) * self.mu * eij
        ###################################################### the reg loss G loss for the new graph  remember to add negative mark
        if if_drop_edge == True:
            # self.reg = 0
            # ## the Laplacian loss
            # x_tilde_deleted_mat = tf.reshape(model.x_tilde_deleted, shape = [self.num_nodes, self.num_nodes])
            # rowsum = tf.reduce_sum(model.adj_ori_dense, axis=0)
            # self.g_delta = rowsum - model.adj_ori_dense
            # temp = tf.matmul(tf.transpose(x_tilde_deleted_mat), self.g_delta)
            # self.reg_mat = tf.matmul(temp, x_tilde_deleted_mat)
            # ###### grab non zero part
            # #self.reg = tf.gather_nd(self.reg_mat, tf.where(self.reg_mat > 0))   # set the bigger then
            # ###### norm version
            # #self.reg = tf.square(tf.norm(self.reg_mat))
            # ###### trace version
            # self.reg = tf.square(tf.trace(self.reg_mat))
            # ####################
            # # self.reg = self.reg * 1e-9
            # self.reg = self.reg
            # #self.reg = tf.reduce_mean(self.reg_mat)
            # self.reg = 1 / (self.reg + 1e-10)
            #
            # ## self.G_comm_loss
            # eij = tf.gather_nd(model.x_tilde_deleted, tf.where(model.x_tilde_deleted > 0))
            # eij = tf.reduce_sum(tf.log(eij))
            # #self.G_comm_loss = (-1)* self.mu * eij + FLAGS.G_KL_r * self.G_comm_loss_KL
            # self.G_comm_loss = (-1) * self.reg * eij
            self.G_comm_loss = self.reg_loss_many_samples(model, self.G_comm_loss)
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



    def reg_loss_many_samples(self, model, G_comm_loss):
        for idx, x_tilde_deleted in enumerate(model.x_tilde_list):
            self.reg = 0
            ## the Laplacian loss
            x_tilde_deleted_mat = tf.reshape(x_tilde_deleted, shape=[self.num_nodes, self.num_nodes])
            rowsum = tf.reduce_sum(model.adj_ori_dense, axis=0)
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

    def reg_loss_many_samples_reward_per(self, model, G_comm_loss):
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
            rowsum = tf.reduce_sum(model.adj_ori_dense, axis=0)
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
            # self.reg = self.reg
            self.reg = tf.log(self.reg + 1)
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
    pass

