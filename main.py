import tensorflow as tf
import random
#from utils import mkdir_p
from utils import randomly_add_edges, randomly_delete_edges
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import normalize
import datetime
from optimizer import OptimizerAE, OptimizerVAE
import numpy as np
import scipy.sparse as sp
import time
import os
# set the random seed
seed = 142   # last random seed is 141           0.703
#random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)
#import sklearn.metrics.normalized_mutual_info_score as normalized_mutual_info_score
from sklearn.metrics import normalized_mutual_info_score
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
from preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges,get_target_nodes_and_comm_labels, construct_feed_dict_trained
from gaegan import gaegan
from optimizer import Optimizergaegan
from gcn.utils import load_data
#import GCN_3L as GCN
from gcn import train_test as GCN
from ops import print_mu, print_mu2

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
# flags.DEFINE_float('learning_rate', 0.005, 'Initial learning rate.')
flags.DEFINE_integer('n_class', 6, 'Number of epochs to train.')
flags.DEFINE_string("target_index_list","10,35", "The index for the target_index")
flags.DEFINE_integer('epochs', 1200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('hidden3', 32, 'Number of units in graphite hidden layers.')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('delete_edge_times', 10, 'sample times for delete K edges. We use this to average the x_tilde(normalized adj) got from generator')

####### for clean gcn training and test
flags.DEFINE_float('gcn_learning_rate', 0.01, 'Initial learning rate.')
#flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('gcn_hidden1', 16, 'Number of units in hidden layer 1.')
#flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('gcn_weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
#flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
###########################
flags.DEFINE_float('dropout', 0.3, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('g_scale_factor', 1- 0.75/2, 'the parametor for generate fake loss')
flags.DEFINE_float('d_scale_factor', 0.25, 'the parametor for discriminator real loss')
flags.DEFINE_float('g_gamma', 1e-06, 'the parametor for generate loss, it has one term with encoder\'s loss')
flags.DEFINE_float('G_KL_r', 0.1, 'The r parameters for the G KL loss')
flags.DEFINE_float('mincut_r', 0.01, 'The r parameters for the cutmin loss orth loss')
flags.DEFINE_float('autoregressive_scalar', 0.2, 'the parametor for graphite generator')
flags.DEFINE_string('model', 'gae_gan', 'Model string.')
flags.DEFINE_string('generator', 'dense', 'Which generator will be used') # the options are "inner_product", "graphite", "graphite_attention", "dense_attention" , "dense"
flags.DEFINE_string('dataset', 'citeseer', 'Dataset string.')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')
# seting from the vae gan
from tensorflow.python.client import device_lib
flags.DEFINE_integer("batch_size" , 64, "batch size")
flags.DEFINE_integer("max_iters" , 600000, "the maxmization epoch")
flags.DEFINE_integer("latent_dim" , 16, "the dim of latent code")
flags.DEFINE_float("learn_rate_init" , 1e-02, "the init of learn rate")
#Please set this num of repeat by the size of your datasets.
flags.DEFINE_integer("repeat", 1000, "the numbers of repeat for your datasets")
flags.DEFINE_string("trained_base_path", '191216023843', "The path for the trained model")
flags.DEFINE_string("trained_our_path", '191215231708', "The path for the trained model")
flags.DEFINE_integer("k", 20, "The k edges to delete")
flags.DEFINE_integer('baseline_target_budget', 5, 'the parametor for graphite generator')
flags.DEFINE_integer("op", 1, "Training or Test")
###############################
if_drop_edge = True
if_save_model = False
# if train the discriminator
if_train_dis = True
restore_trained_our = False
showed_target_idx = 0   # the target index group of targets you want to show
###################################
### read and process the graph
model_str = FLAGS.model
dataset_str = FLAGS.dataset
# Load data
# _A_obs, _X_obs, _z_obs = utils.load_npz('data/citeseer.npz')
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data("citeseer")

# _A_obs = _A_obs + _A_obs.T #变GCN_ori as GCN
# _A_obs[_A_obs > 1] = 1
# adj = _A_obs

adj_norm, adj_norm_sparse = preprocess_graph(adj)

#_K = _z_obs.max()+1 #类别个数
_K = y_train.shape[1]
features_normlize = normalize(features.tocsr(), axis=0, norm='max')
features = sp.csr_matrix(features_normlize)

#add comm_label this time to get the good accuracy
# node_labels = np.eye(_K)[_z_obs] #把标签转化为one-hot


# Store original adjacency matrix (without diagonal entries) for later


# adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
# adj = adj_train
if FLAGS.features == 0:
    features = sp.identity(features.shape[0])  # featureless
# Some preprocessing

placeholders = {
    'features': tf.sparse_placeholder(tf.float32, name="ph_features"),
    'adj': tf.sparse_placeholder(tf.float32, name="ph_adj"),
    'adj_orig': tf.sparse_placeholder(tf.float32, name="ph_orig"),
    'dropout': tf.placeholder_with_default(0., shape=(), name="ph_dropout"),
    # 'node_labels': tf.placeholder(tf.float32, name="ph_node_labels"),
    # 'node_ids': tf.placeholder(tf.float32, name="ph_node_ids")
}

num_nodes = adj.shape[0]
features_csr = features
features_csr = features_csr.astype('float32')
features = sparse_to_tuple(features.tocoo())
num_features = features[2][1]
features_nonzero = features[1].shape[0]
n_class = _K

# seed = 68
unlabeled_share = 0.8  # the propotion for test
val_share = 0.1        # the propotion for validation
train_share = 1 - unlabeled_share - val_share # the proportion for trainingr
gpu_id = 1
# np.random.seed(seed)
# split_train, split_val, split_unlabeled = utils.train_val_test_split_tabular(np.arange(num_nodes),
#                                                                        train_size=train_share,
#                                                                        val_size=val_share,
#                                                                        test_size=unlabeled_share,
#                                                                        stratify=_z_obs)

# Create model

    #session part
cost_val = []
acc_val = []

cost_val = []
acc_val = []
val_roc_score = []


def get_roc_score(edges_pos, edges_neg,feed_dict,sess, model, emb=None):
    if emb is None:
        feed_dict.update({placeholders['dropout']: 0})
        adj_rec = sess.run(model.x_tilde, feed_dict=feed_dict)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    preds = []
    pos = []
    for e in edges_pos:
        #preds.append(sigmoid(adj_rec[e[0], e[1]]))
        preds.append(adj_rec[e[0], e[1]])
        pos.append(adj_orig[e[0], e[1]])
    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])
    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    return roc_score, ap_score

def get_new_adj(feed_dict, sess, model):
    new_adj = model.new_adj_without_norm.eval(session=sess, feed_dict=feed_dict)
    return new_adj

def train_dis_base():
    tf.reset_default_graph()
    if_drop_edge = False
    ## set the placeholde
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32, name= "ph_features"),
        'adj': tf.sparse_placeholder(tf.float32,name= "ph_adj"),
        'adj_orig': tf.sparse_placeholder(tf.float32, name = "ph_orig"),
        'dropout': tf.placeholder_with_default(0., shape=(), name = "ph_dropout"),
        #'comm_label': tf.placeholder(tf.float32, name = "ph_comm_label")
    }

    ## set the checkpoints path for base
    checkpoints_dir_base = "./checkpoints/base"
    current_time = datetime.datetime.now().strftime("%y%m%d%H%M%S")

    checkpoints_dir = os.path.join(checkpoints_dir_base, current_time, current_time)
    global_steps = tf.get_variable('global_step', trainable=False, initializer=0)
    new_learning_rate = tf.train.exponential_decay(FLAGS.learn_rate_init, global_step=global_steps, decay_steps=1000,
                                                   decay_rate=0.98)
    # build models
    model = None
    if model_str == "gae_gan":
        model = gaegan(placeholders, num_features, num_nodes, features_nonzero, new_learning_rate,if_drop_edge)
        model.build_model()
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    opt = 0
    # Optimizer
    with tf.name_scope('optimizer'):
        if model_str == 'gae_gan':
            opt = Optimizergaegan(preds=tf.reshape(model.x_tilde, [-1]),
                                  labels=tf.reshape(
                                      tf.sparse_tensor_to_dense(placeholders['adj_orig'], validate_indices=False),
                                      [-1]),
                                  #comm_label=placeholders["comm_label"],
                                  model=model,
                                  num_nodes=num_nodes,
                                  pos_weight=pos_weight,
                                  norm=norm,
                                  global_step=global_steps,
                                  new_learning_rate=new_learning_rate,
                                  if_drop_edge = if_drop_edge
                                  )
    # init the sess
    sess_bass = tf.Session()
    sess_bass.run(tf.global_variables_initializer())
    #var_list = tf.global_variables()
    #var_list = [var for var in var_list if "discriminate" in var.name]   # check if we need to do this
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)
    if if_save_model:
        os.mkdir(os.path.join(checkpoints_dir_base, current_time))
        saver.save(sess_bass, checkpoints_dir)  # save the graph
    feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    for epoch in range(FLAGS.epochs):
        t = time.time()
        # Construct feed dictionary
        # run D optimizer
        sess_bass.run(opt.D_min_op, feed_dict=feed_dict)   # here we only run the base model
        # # Comoptpute average lossgaegan
        #avg_acc_D = sess_bass.run(opt.D_accuracy, feed_dict)
        if epoch % 50 == 0:
            print("Epoch:", '%04d' % (epoch + 1),
                  "time=", "{:.5f}".format(time.time() - t))
            D_loss, new_learn_rate_value = sess_bass.run([opt.D_mincut_loss_test, new_learning_rate],
                                                            feed_dict=feed_dict)
            #new_adj = model.x_tilde_output.eval(session=sess_bass, feed_dict=feed_dict)
            #temp_pred = new_adj.reshape(-1)
            #temp_ori = adj_label_sparse.todense().A.reshape(-1)
            #mutual_info = normalized_mutual_info_score(temp_pred, temp_ori)
            print("Step %d: D: loss = %.7f , LR=%.7f" % (
            epoch, D_loss, new_learn_rate_value))
            ## check the target attack results
            pred_dis_res = model.realD_tilde.eval(session=sess_bass, feed_dict=feed_dict)

            # for targets in target_list:
            #     for target in targets:
            #         pred_target = pred_dis_res[target, :]
            #         print("The target %d:" % (target))
            #         print(pred_target
            # for target in targets:
            #     pred_target = pred_dis_res[target, :]
            #     print("The target %d:" % (target))
            #     print(pred_target)
            # print("#" * 30)
        if (epoch % 200 == 1) and (if_save_model):
            saver.save(sess_bass, checkpoints_dir, global_step=epoch, write_meta_graph=False)
    saver.save(sess_bass, checkpoints_dir, global_step=FLAGS.epochs, write_meta_graph=False)
    print("Optimization Finished!")
    # new_adj = get_new_adj(feed_dict, sess_bass, model)
    # np.save("./data/hinton/hinton_new_adj_48_0815.npy", new_adj)
    ##### The final results ####
    feed_dict.update({placeholders['dropout']: 0})
    pred_dis_res = model.realD_tilde.eval(session=sess_bass, feed_dict=feed_dict)  # the prediction for the base model
    print("*" * 30)
    print("the final results for base model:\n")
    # for targets in target_list:
    #     for target in targets:
    #         pred_target = pred_dis_res[target, :]
    #         print("The target %d:" % (target))
    #         print(pred_target)
    # new_adj = get_new_adj(feed_dict, sess_bass, model)
    # temp_pred = new_adj.reshape(-1)
    # temp_ori = adj_norm_sparse.todense().A.reshape(-1)
    # mutual_info = normalized_mutual_info_score(temp_pred, temp_ori)
    # #print(f"The mutual informations is {mutual_info}")
    # print("The mutual informations is:")
    # print(mutual_info)
    return

# load the trained base model to check the results
# if_ori is if we use the adj original or the D-AD-. If the input is adj original we should convert it into normalized graph
def trained_dis_base(adj_norm,new_adj, if_ori):
    tf.reset_default_graph()
    if_drop_edge = False
    ## set the checkpoints path for base
    checkpoints_dir_base = "./checkpoints/base"
    checkpoints_dir = os.path.join(checkpoints_dir_base, FLAGS.trained_base_path)
    checkpoints_dir_meta = os.path.join(checkpoints_dir_base, FLAGS.trained_base_path, FLAGS.trained_base_path + ".meta")
    #global_steps = tf.get_variable('global_step', trainable=False, initializer=0)
    with tf.Session() as sess:
        #saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
        saver = tf.train.import_meta_graph(checkpoints_dir_meta)
        graph = tf.get_default_graph()
        ## set the placeholde
        placeholders = {
            'features': {"indices":graph.get_tensor_by_name("ph_features/indices:0"),
                         "values" :graph.get_tensor_by_name("ph_features/values:0"),
                         "shape": graph.get_tensor_by_name("ph_features/shape:0")
                         },
            'adj': {"indices":graph.get_tensor_by_name("ph_adj/indices:0"),
                         "values" :graph.get_tensor_by_name("ph_adj/values:0"),
                         "shape": graph.get_tensor_by_name("ph_adj/shape:0")
                         },
            'adj_orig': {"indices":graph.get_tensor_by_name("ph_orig/indices:0"),
                         "values" :graph.get_tensor_by_name("ph_orig/values:0"),
                         "shape": graph.get_tensor_by_name("ph_orig/shape:0")
                         },
            'dropout': graph.get_tensor_by_name("ph_dropout:0"),
            #'comm_label': graph.get_tensor_by_name("ph_comm_label:0")
        }
        saver.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
        adj_norm_trained = adj_norm
        if not if_ori:
            mutual_info = normalized_mutual_info_score(new_adj.reshape(-1), adj_label_sparse.todense().A.reshape(-1))
            new_adj_nodiag = new_adj - np.diag(np.diag(new_adj))
            adj_norm_trained, adj_norm_sparse_trained = preprocess_graph(new_adj_nodiag)
            new_adj = sparse_to_tuple(sp.csr_matrix(new_adj))
            print("The mutual information is:%f"%(mutual_info))
        feed_dict = construct_feed_dict_trained(adj_norm_trained, new_adj, features, placeholders)
        # feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        feed_dict.update({placeholders['dropout']: 0})
        realD_tilde = graph.get_tensor_by_name("discriminate_1/Softmax_1:0")
        pred_dis_res = realD_tilde.eval(session=sess, feed_dict=feed_dict)

    return

# Train model
def train():
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]),
                                        shape=adj_orig.shape)  # delete self loop
    adj_orig.eliminate_zeros()

    adj_new = randomly_add_edges(adj_orig, k=FLAGS.k)

    # train GCN first
    # sizes = [FLAGS.gcn_hidden1, FLAGS.gcn_hidden2, n_class]
    # surrogate_model = GCN.GCN(sizes, adj_norm_sparse_csr, features_csr, with_relu=True, name="surrogate", gpu_id=gpu_id)
    # surrogate_model.train(adj_norm_sparse_csr, split_train, split_val, node_labels)
    # ori_acc = surrogate_model.test(split_unlabeled, node_labels, adj_norm_sparse_csr)
    testacc_clean, valid_acc_clean = GCN.run(FLAGS.dataset, adj_orig, name = "clean")
    testacc, valid_acc = GCN.run(FLAGS.dataset, adj_new, name = "original")
    adj_norm, adj_norm_sparse = preprocess_graph(adj_new)
    adj_norm_sparse_csr = adj_norm_sparse.tocsr()
    adj_label = adj_new + sp.eye(adj.shape[0])
    adj_label_sparse = adj_label
    adj_label = sparse_to_tuple(adj_label)

    if_drop_edge = True
    ## set the checkpoint path
    checkpoints_dir_base = "./checkpoints"
    current_time = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    checkpoints_dir = os.path.join(checkpoints_dir_base, current_time, current_time)
    ############
    global_steps = tf.get_variable('global_step', trainable=False, initializer=0)
    new_learning_rate = tf.train.exponential_decay(FLAGS.learn_rate_init, global_step=global_steps, decay_steps=10000,
                                                   decay_rate=0.98)
    new_learn_rate_value = FLAGS.learn_rate_init
    ## set the placeholders

    placeholders = {
        'features': tf.sparse_placeholder(tf.float32, name= "ph_features"),
        'adj': tf.sparse_placeholder(tf.float32,name= "ph_adj"),
        'adj_orig': tf.sparse_placeholder(tf.float32, name = "ph_orig"),
        'dropout': tf.placeholder_with_default(0., shape=(), name = "ph_dropout"),
        # 'node_labels': tf.placeholder(tf.float32, name = "ph_node_labels"),
        # 'node_ids' : tf.placeholder(tf.float32, name = "ph_node_ids")
    }
    # build models
    model = None
    if model_str == "gae_gan":
        model = gaegan(placeholders, num_features, num_nodes, features_nonzero, new_learning_rate)
        model.build_model()
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    opt = 0
    # Optimizer
    with tf.name_scope('optimizer'):
        if model_str == 'gae_gan':
            opt = Optimizergaegan(preds=tf.reshape(model.x_tilde, [-1]),
                                  labels=tf.reshape(
                                      tf.sparse_tensor_to_dense(placeholders['adj_orig'], validate_indices=False),
                                      [-1]),
                                  #comm_label=placeholders["comm_label"],
                                  model=model,
                                  num_nodes=num_nodes,
                                  pos_weight=pos_weight,
                                  norm=norm,
                                  global_step=global_steps,
                                  new_learning_rate = new_learning_rate
                                  )
    # init the sess
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = ""
    var_list = tf.global_variables()
    var_list = [var for var in var_list if ("encoder"  in var.name) or ('generate' in var.name)]
    saver = tf.train.Saver(var_list, max_to_keep=10)
    if if_save_model:
        os.mkdir(os.path.join(checkpoints_dir_base, current_time))
        saver.save(sess, checkpoints_dir)  # save the graph

    if restore_trained_our:
        checkpoints_dir_our = "./checkpoints"
        checkpoints_dir_our = os.path.join(checkpoints_dir_our, FLAGS.trained_our_path)
        # checkpoints_dir_meta = os.path.join(checkpoints_dir_base, FLAGS.trained_our_path,
        #                                     FLAGS.trained_our_path + ".meta")
        #saver.restore(sess,tf.train.latest_checkpoint(checkpoints_dir_our))
        saver.restore(sess, os.path.join("./checkpoints","191215231708","191215231708-1601"))
        print("model_load_successfully")
    # else:  # if not restore the original then restore the base dis one.
    #     checkpoints_dir_base = os.path.join("./checkpoints/base", FLAGS.trained_base_path)
    #     saver.restore(sess, tf.train.latest_checkpoint(checkpoints_dir_base))

    feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    # pred_dis_res = model.vaeD_tilde.eval(session=sess, feed_dict=feed_dict)

    #### save new_adj without norm#############
    if restore_trained_our:
        modified_adj =  get_new_adj(feed_dict,sess, model)
        modified_adj = sp.csr_matrix(modified_adj)
        sp.save_npz("transfer_new/transfer_1216_1/qq_5000_gaegan_new.npz", modified_adj)
        sp.save_npz("transfer_new/transfer_1216_1/qq_5000_gaegan_ori.npz", adj_new)
        print("save the loaded adj")
    # print("before training generator")
    #####################################################

    #####################################################
    G_loss_min = 1000
    for epoch in range(FLAGS.epochs):
        t = time.time()
        # run Encoder's optimizer
        #sess.run(opt.encoder_min_op, feed_dict=feed_dict)
        # run G optimizer  on trained model
        if restore_trained_our:
            sess.run(opt.G_min_op, feed_dict=feed_dict)
        else: # it is the new model
            if epoch < FLAGS.epochs:
                sess.run(opt.G_min_op, feed_dict=feed_dict)
            #
        ##
        ##
        if epoch % 50 == 0:
            print("Epoch:", '%04d' % (epoch + 1),
                  "time=", "{:.5f}".format(time.time() - t))
            G_loss, laplacian_para,new_learn_rate_value = sess.run([opt.G_comm_loss,opt.reg,new_learning_rate],feed_dict=feed_dict)
            #new_adj = get_new_adj(feed_dict, sess, model)
            new_adj = model.new_adj_output.eval(session = sess, feed_dict = feed_dict)
            temp_pred = new_adj.reshape(-1)
            #temp_ori = adj_norm_sparse.todense().A.reshape(-1)
            temp_ori = adj_label_sparse.todense().A.reshape(-1)
            mutual_info = normalized_mutual_info_score(temp_pred, temp_ori)
            print("Step: %d,G: loss=%.7f ,Lap_para: %f  ,info_score = %.6f, LR=%.7f" % (epoch, G_loss,laplacian_para, mutual_info,new_learn_rate_value))
            ## here is the debug part of the model#################################
            laplacian_mat, reg_trace, reg_log, reward_ratio = sess.run([opt.reg_mat, opt.reg_trace, opt.reg_log, opt.new_percent_softmax], feed_dict=feed_dict)
            print("lap_mat is:")
            print(np.diag(laplacian_mat))
            print("reg_trace is:")
            print(reg_trace)
            print("reg_log is:")
            print(reg_log)
            print("reward_percentage")
            print(reward_ratio)
            ##########################################
            #';# check the D_loss_min
            if (G_loss < G_loss_min) and (epoch > 1000) and (if_save_model):
                saver.save(sess, checkpoints_dir, global_step=epoch, write_meta_graph=False)
                print("min G_loss new")
            if G_loss < G_loss_min:
                G_loss_min = G_loss

        if (epoch % 200 ==1) and if_save_model:
            saver.save(sess,checkpoints_dir, global_step = epoch, write_meta_graph = False)
            print("Epoch:", '%04d' % (epoch + 1),
                  "time=", "{:.5f}".format(time.time() - t))
    saver.save(sess, checkpoints_dir, global_step=FLAGS.epochs, write_meta_graph=True)
    print("Optimization Finished!")
    feed_dict.update({placeholders['dropout']: 0})
    new_adj = get_new_adj(feed_dict,sess, model)
    new_adj = new_adj - np.diag(np.diag(new_adj))
    new_adj_sparse = sp.csr_matrix(new_adj)
    print((abs(new_adj_sparse - new_adj_sparse.T) > 1e-10).nnz == 0)
    # new_adj_norm, new_adj_norm_sparse = preprocess_graph(new_adj)
    # new_adj_norm_sparse_csr = new_adj_norm_sparse.tocsr()
    # modified_model = GCN.GCN(sizes, new_adj_norm_sparse_csr, features_csr, with_relu=True, name="surrogate", gpu_id=gpu_id)
    # modified_model.train(new_adj_norm_sparse_csr, split_train, split_val, node_labels)
    # modified_acc = modified_model.test(split_unlabeled, node_labels, new_adj_norm_sparse_csr)
    testacc_new, valid_acc_new = GCN.run(FLAGS.dataset, new_adj_sparse, name = "modified")
    new_adj = get_new_adj(feed_dict, sess, model)
    new_adj = new_adj - np.diag(np.diag(new_adj))
    new_adj_sparse = sp.csr_matrix(new_adj)
    testacc_new2, valid_acc_new = GCN.run(FLAGS.dataset, new_adj_sparse, name="modified")
    new_adj = get_new_adj(feed_dict, sess, model)
    new_adj = new_adj - np.diag(np.diag(new_adj))
    new_adj_sparse = sp.csr_matrix(new_adj)
    testacc_new3, valid_acc_new = GCN.run(FLAGS.dataset, new_adj_sparse, name="modified")
    #np.save("./data/hinton/hinton_new_adj_48_0815.npy", new_adj)
    #roc_score, ap_score = get_roc_score(test_edges, test_edges_false,feed_dict, sess, model)
    ##### The final results ####
    print("*" * 30)
    print("the final results:\n")
    print("The original acc is: ")
    print(testacc)
    print("*#"* 15)
    print("The modified acc is : ")
    print(testacc_new)
    print("*#" * 15)
    print("The modified acc is : ")
    print(testacc_new2)
    print("*#" * 15)
    print("The modified acc is : ")
    print(testacc_new3)
    return new_adj, testacc, testacc_new, testacc_new2, testacc_new3
## delete edges between the targets and 1add some
def base_line():
    target_budget = np.random.choice(len(target_list), FLAGS.baseline_target_budget, replace = False)
    target_budget = target_list[target_budget]
    adj_base_sparse = adj_label_sparse
    edge_list = []
    for targets in target_budget:
        for i in range(len(targets)):
            for j in range(i+1,len(targets)):
                if adj_orig[targets[i],targets[j]] == 1:
                    edge_list.append([targets[i],targets[j]])
                elif adj_orig[targets[j],targets[i]] == 1:
                    edge_list.append([targets[i],targets[j]])
    ## selected delete edges
    edge_list = np.array(edge_list)
    if edge_list !=[]:
        selected_list_idx = np.random.choice(len(edge_list), min(int(FLAGS.k // 2), len(edge_list)), replace=False)
        selected_list = edge_list[selected_list_idx]
        for pair in selected_list:
            adj_base_sparse[pair[0], pair[1]] = 0
            adj_base_sparse[pair[1], pair[0]] = 0

    ## add some edges on random
    modified_num = FLAGS.baseline_target_budget
    target_edge_list = []
    edge_list = []
    num_nodes = adj_orig.shape[0]
    for targets in target_budget:
        for i in range(modified_num):
            for target in targets:
                random_node = np.random.choice(num_nodes, 1)
                while (adj_orig[target, random_node] ==1) or (random_node == target):
                    random_node = np.random.choice(num_nodes,1)
                if [target, random_node] not in target_edge_list:
                    target_edge_list.append([target, random_node])

    ## selected add edges
    selected_list_idx = np.random.choice(len(target_edge_list), FLAGS.k - int(FLAGS.k // 2), replace=False)
    target_edge_list = np.array(target_edge_list)
    selected_list = target_edge_list[selected_list_idx]
    for pair in selected_list:
        adj_base_sparse[pair[0], pair[1]] = 1
        adj_base_sparse[pair[1], pair[0]] = 1
    adj_base_new_dense= adj_base_sparse.todense().A
    print("the clean adj")
    trained_dis_base(adj_norm, adj_label, if_ori=True)
    print("the baseline1 ")
    trained_dis_base(adj_norm, adj_base_new_dense, if_ori=False)
    #trained_dis_base(adj_norm, adj_base_sparse, if_ori=True)
    ## save the baseline1 transfer adj
    sp.save_npz("transfer_new/transfer_1216_1/qq_5000_base1_new.npz", adj_base_sparse)
    sp.save_npz("transfer_new/transfer_1216_1/qq_5000_base1_ori.npz", adj_orig)
    print("save the loaded model")
    return

def base_line_add():
    edge_list = []
    for targets in target_list:
        for i in range(len(targets)):
            for j in range(i+1,len(targets)):
                if adj_orig[targets[i],targets[j]] == 0:
                    edge_list.append([targets[i],targets[j]])
                elif adj_orig[targets[j],targets[i]] == 0:
                    edge_list.append([targets[i],targets[j]])
    ## selected delete edges
    edge_list = np.array(edge_list)
    selected_list_idx = np.random.choice(len(edge_list), FLAGS.k, replace=False)
    selected_list = edge_list[selected_list_idx]
    adj_base_sparse = adj_label_sparse
    for pair in selected_list:
        adj_base_sparse[pair[0], pair[1]] = 1
        adj_base_sparse[pair[1], pair[0]] = 1
    adj_base_new_dense= adj_base_sparse.todense().A
    print("the clean")
    trained_dis_base(adj_norm, adj_label, if_ori=True)
    print("The baseline1")
    trained_dis_base(adj_norm, adj_base_new_dense, if_ori=False)
    return

#target 和随机其他点加边  the degree one
def base_line2():
    add_edge_list = []
    delete_edge_list = []
    adj_orig_dense = adj_orig.todense().A
    num_nodes = adj_orig.shape[0]
    Degree_nodes = np.array(adj_orig.sum(1))
    Degree_nodes = Degree_nodes.reshape(-1)
    target_flat = target_list.reshape(-1)
    target_indexes = np.zeros(len(Degree_nodes))
    target_indexes[target_flat] = 1
    degree_targets = Degree_nodes * target_indexes
    max_deg_indexes = np.argsort(degree_targets)[-1*int(FLAGS.k / 2):]
    target_indexes = np.zeros(len(Degree_nodes))
    target_indexes[target_flat] = 1
    degree_targets = Degree_nodes * target_indexes
    degree_targets[target_indexes == 0] = np.max(Degree_nodes)
    degree_targets[degree_targets == 0] = np.max(Degree_nodes)  # we dont think about the node who does not have neighors
    min_deg_indexes = np.argsort(degree_targets)[:(FLAGS.k - int(FLAGS.k / 2))]
    ######################
    for i in max_deg_indexes:  # add one edge for not neighbor
        neighbors_idx = np.array(adj_orig[i,:].todense()).reshape(-1)
        neighbor_indexes = np.ones(len(Degree_nodes))
        neighbor_indexes[neighbors_idx == 1] = 0
        neighbor_indexes[i] = 0
        neighbors_degree = Degree_nodes * neighbor_indexes
        other_idx = np.argmax(neighbors_degree)
        add_edge_list.append([i,other_idx])
    for i in min_deg_indexes:   # delete one edge for min neighbor
        neighbors_idx = np.array(adj_orig[i, :].todense()).reshape(-1)
        neighbor_indexes = np.zeros(len(Degree_nodes))
        neighbor_indexes[neighbors_idx == 1] = 1
        neighbors_degree = Degree_nodes * neighbor_indexes
        neighbors_degree[neighbors_idx == 0] = np.max(Degree_nodes)
        other_idx = np.argmin(neighbors_degree)
        delete_edge_list.append([i, other_idx])
    a = 1
    adj_base_sparse = adj_label_sparse
    for pair in add_edge_list:
        adj_base_sparse[pair[0], pair[1]] = 1
        adj_base_sparse[pair[1], pair[0]] = 1
    for pair in delete_edge_list:
        adj_base_sparse[pair[0], pair[1]] = 0
        adj_base_sparse[pair[1], pair[0]] = 0
    adj_base_new_dense= adj_base_sparse.todense().A
    print("the clean adj")
    trained_dis_base(adj_norm, adj_label, if_ori=True)
    print("the baseline2 ")
    trained_dis_base(adj_norm, adj_base_new_dense, if_ori=False)

    ## save the baseline1 transfer adj
    sp.save_npz("transfer_new/transfer_1216_1/qq_5000_base2_new.npz", adj_base_sparse)
    sp.save_npz("transfer_new/transfer_1216_1/qq_5000_base2_ori.npz", adj_orig)
    print("save the loaded model")
    return

#target 和随机其他点加边
def base_line3():
    target_budget = np.random.choice(len(target_list), FLAGS.baseline_target_budget, replace = False)
    target_budget = target_list[target_budget]
    modify_times = FLAGS.baseline_target_budget
    adj_base_sparse = adj_label_sparse
    target_edge_list = []
    edge_list = []
    num_nodes = adj_orig.shape[0]
    for targets in target_budget:
        for i in range(modify_times):
            for target in targets:
                random_node = np.random.choice(num_nodes, 1)
                while (adj_orig[target, random_node] ==1) or (random_node == target):
                    random_node = np.random.choice(num_nodes,1)
                if [target, random_node[0]] not in target_edge_list:
                    target_edge_list.append([target, random_node[0]])

    ## selected delete edges
    selected_list_idx = np.random.choice(len(target_edge_list), FLAGS.k - int(FLAGS.k // 2), replace=False)
    target_edge_list = np.array(target_edge_list)
    selected_list = target_edge_list[selected_list_idx]
    for pair in selected_list:
        adj_base_sparse[pair[0], pair[1]] = 1
        adj_base_sparse[pair[1], pair[0]] = 1

    ## delete random list from target to others
    target_edge_list = []
    edge_list = []
    num_nodes = adj_orig.shape[0]
    for targets in target_budget:
        for i in range(modify_times):
            for target in targets:
                random_node = np.random.choice(num_nodes, 1)
                temp_idx = 0
                while (adj_orig[target, random_node] == 0) or (random_node == target):
                    random_node = np.random.choice(num_nodes, 1)
                    temp_idx += 1
                    if temp_idx >2000:
                        break
                if [target, random_node] not in target_edge_list:
                    target_edge_list.append([target, random_node])

    ## selected delete edges
    selected_list_idx = np.random.choice(len(target_edge_list), int(FLAGS.k // 2), replace=False)
    target_edge_list = np.array(target_edge_list)
    selected_list = target_edge_list[selected_list_idx]
    for pair in selected_list:
        adj_base_sparse[pair[0], pair[1]] = 0
        adj_base_sparse[pair[1], pair[0]] = 0
    adj_base_new_dense= adj_base_sparse.todense().A
    print("The clean one")
    trained_dis_base(adj_norm, adj_label, if_ori=True)
    print("the baseline3")
    trained_dis_base(adj_norm, adj_base_new_dense, if_ori=False)

    ## save the baseline1 transfer adj
    sp.save_npz("transfer_new/transfer_1216_1/qq_5000_base3_new.npz", adj_base_sparse)
    sp.save_npz("transfer_new/transfer_1216_1/qq_5000_base3_ori.npz", adj_orig)
    print("save the loaded model")
    return


def test(saver,adj,features, meta_dir, checkpoints_dir):
    adj_norm, adj_norm_sparse = preprocess_graph(adj)
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=())
    }

    num_nodes = adj.shape[0]
    features = sparse_to_tuple(features.tocoo())
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]
    feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    # Create model
    saver = tf.train.Saver(max_to_keep=10)
    model = None
    if model_str == "gae_gan":
        model = gaegan(placeholders, num_features, num_nodes, features_nonzero)
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    global_steps = tf.get_variable(0, name="globals")
    opt = 0
    # Optimizer
    with tf.name_scope('optimizer'):
        if model_str == 'gae_gan':
            opt = Optimizergaegan(preds=model.x_tilde,
                                  labels=tf.reshape(
                                      tf.sparse_tensor_to_dense(placeholders['adj_orig'], validate_indices=False),
                                      [-1]),
                                  model=model,
                                  num_nodes=num_nodes,
                                  pos_weight=pos_weight,
                                  norm=norm,
                                  global_step=global_steps
                                  )

        # session part
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    cost_val = []
    acc_val = []
    # load network
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(meta_dir)
        saver.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
        sess.run()
        new_adj = get_new_adj(feed_dict)
    return new_adj

FLAGS = flags.FLAGS
if __name__ == "__main__":
    #train_dis_base()
    current_time = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    with open("results/results_%d_%s.txt"%(FLAGS.k, current_time), 'w+') as f_out:
        for i in range(5):
            new_adj, testacc, testaccnew1, testaccnew2, testaccnew3 = train()
            # testacc = 1.01
            # testaccnew1 = 1.01
            # testaccnew2 = 1.01
            # testaccnew3 = 1.01
            f_out.write(str(testacc)+ ' '+str(testaccnew1)+ ' '+str(testaccnew2)+ ' '+str(testaccnew3)+"\n")

    # print("The original base model")
    #trained_dis_base(adj_norm, adj_label, if_ori = True)  #
    # print("The modified model base model")
    #trained_dis_base(adj_norm, new_adj, if_ori=False)
    #print("The modified model base model using x_tilde")
    #trained_dis_base(adj_norm, x_tilde_out, if_ori=False)
    # print("finish")

    # base_line()
    # base_line2()
    # base_line3()
