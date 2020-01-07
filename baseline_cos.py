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
from scipy.sparse.linalg import norm as spnorm
import time
flags = tf.app.flags
FLAGS = flags.FLAGS
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# set the random seed
seed = 142   # last random seed is 141           0.703
#random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)
#import sklearn.metrics.normalized_mutual_info_score as normalized_mutual_info_score
from sklearn.metrics import normalized_mutual_info_score
from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())
from preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges,get_target_nodes_and_comm_labels, construct_feed_dict_trained
from gaegan import gaegan
from optimizer import Optimizergaegan
from gcn.utils import load_data
#import GCN_3L as GCN
from gcn import train_test as GCN
from ops import print_mu, print_mu2
# Settings
# flags.DEFINE_float('learning_rate', 0.005, 'Initial learning rate.')
flags.DEFINE_string("function_name", "add", "The function of the baseline. 'add' or 'delete'")
flags.DEFINE_string("gpu_id", '0', "The gpu id used for baseline training ")
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
flags.DEFINE_integer("k", 50, "The k edges to delete")
flags.DEFINE_integer('baseline_target_budget', 5, 'the parametor for graphite generator')
flags.DEFINE_integer("op", 1, "Training or Test")
################################
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
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)   # delete self loop
adj_orig.eliminate_zeros()
adj_norm, adj_norm_sparse = preprocess_graph(adj)

#_K = _z_obs.max()+1 #类别个数
_K = y_train.shape[1]
features_normlize = normalize(features.tocsr(), axis=0, norm='max')
features = sp.csr_matrix(features_normlize)

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
num_features = features[2][1]  # shape
features_nonzero = features[1].shape[0]   # the value
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

adj_label = adj_orig + sp.eye(adj.shape[0])
adj_label_sparse = adj_label
adj_label = sparse_to_tuple(adj_label)

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


def baseline_delete():
    adj_new = randomly_add_edges(adj_orig, k = FLAGS.k)
    testacc_clean, valid_acc_clean =  GCN.run(FLAGS.dataset, adj_orig, name="clean")
    testacc, valid_acc = GCN.run(FLAGS.dataset, adj_new, name="original")
    cos = features_csr.dot(features_csr.transpose())
    norm = spnorm(features_csr, axis=1)
    norm = norm[:,np.newaxis]
    norm_mat = norm.dot(norm.T)
    cos = cos / norm_mat
    normalize_cos = 0.5 + 0.5 * cos
    normalize_cos = np.array(normalize_cos)
    one_mat = np.ones([num_nodes, num_nodes])
    adj_new_dense = adj_new.todense()
    one_mat[np.triu(adj_new_dense, k=0) > 0] = normalize_cos[np.triu(adj_new_dense,k=0) > 0]
    one_mat = one_mat.flatten()
    one_mat[np.isnan(one_mat)] = 1
    deleted_idx = np.argsort(one_mat)
    deleted_idx = deleted_idx[:FLAGS.k]
    row_idx =deleted_idx %num_nodes
    col_idx = deleted_idx // num_nodes
    for idx in range(len(row_idx)):
        adj_new[row_idx, col_idx] = 0
        adj_new[col_idx, row_idx] = 0
    testacc_new, valid_acc_new = GCN.run(FLAGS.dataset, adj_new, name="new1")
    testacc_new2, valid_acc_new = GCN.run(FLAGS.dataset, adj_new, name="new2")
    testacc_new3, valid_acc_new = GCN.run(FLAGS.dataset, adj_new, name="new3")
    print("**#" * 10)
    print("clean one")
    print(testacc_clean)
    print("**#"*10)
    print("noised one")
    print(testacc)
    print("new one")
    print(testacc_new)
    print("new two")
    print(testacc_new2)
    print("new three")
    print(testacc_new3)
    print("**#" * 10)
    return testacc_clean, testacc, testacc_new, testacc_new2, testacc_new3

def baseline_add():
    adj_new = randomly_delete_edges(adj_orig, k  = FLAGS.k)
    testacc_clean, valid_acc_clean = GCN.run(FLAGS.dataset, adj_orig, name="clean")
    testacc, valid_acc = GCN.run(FLAGS.dataset, adj_new, name="original")
    cos = features_csr.dot(features_csr.transpose())
    norm = spnorm(features_csr, axis=1)
    norm = norm[:,np.newaxis]
    norm_mat = norm.dot(norm.T)
    cos = cos / norm_mat
    normalize_cos = 0.5 + 0.5 * cos
    normalize_cos = np.array(normalize_cos)
    zero_mat = np.zeros([num_nodes, num_nodes])
    adj_new_dense = adj_new.todense()
    flag_adj = np.triu(np.ones([num_nodes, num_nodes]), k=1) - np.triu(adj_new_dense, k=1)
    zero_mat[flag_adj > 0] = normalize_cos[flag_adj > 0]
    one_mat = zero_mat.flatten()
    one_mat[np.isnan(one_mat)] = 0
    add_idx = np.argsort(one_mat)
    add_idx = add_idx[-FLAGS.k:]
    row_idx =add_idx %num_nodes
    col_idx = add_idx // num_nodes
    for idx in range(len(row_idx)):
        adj_new[row_idx, col_idx] = 1
        adj_new[col_idx, row_idx] = 1
    testacc_new, valid_acc_new = GCN.run(FLAGS.dataset, adj_new, name="new1")
    testacc_new2, valid_acc_new = GCN.run(FLAGS.dataset, adj_new, name="new2")
    testacc_new3, valid_acc_new = GCN.run(FLAGS.dataset, adj_new, name="new3")
    print("**#" * 10)
    print("clean one")
    print(testacc_clean)
    print("**#"*10)
    print("original one")
    print(testacc)
    print("new one")
    print(testacc_new)
    print("new two")
    print(testacc_new2)
    print("new three")
    print(testacc_new3)
    print("**#" * 10)
    return testacc_clean, testacc, testacc_new, testacc_new2, testacc_new3



# Train model
def train():
    # train GCN first
    adj_norm_sparse_csr = adj_norm_sparse.tocsr()
    # sizes = [FLAGS.gcn_hidden1, FLAGS.gcn_hidden2, n_class]
    # surrogate_model = GCN.GCN(sizes, adj_norm_sparse_csr, features_csr, with_relu=True, name="surrogate", gpu_id=gpu_id)
    # surrogate_model.train(adj_norm_sparse_csr, split_train, split_val, node_labels)
    # ori_acc = surrogate_model.test(split_unlabeled, node_labels, adj_norm_sparse_csr)
    testacc, valid_acc = GCN.run(FLAGS.dataset, adj_orig, name = "original")
    cos = sp.dot(features_csr.transpose(), features_csr)
    norm = spnorm(features_csr, axis = 1)
    norm_mat = norm.dot(norm.transpose())

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
        sp.save_npz("transfer_new/transfer_1216_1/qq_5000_gaegan_ori.npz", adj_orig)
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
if __name__ == "__main__":
    #train_dis_base()
    function_name = FLAGS.function_name  ## add delete
    current_time = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    with open("results/baseline/results_%d_%s_%s.txt"%(FLAGS.k, current_time, function_name), 'w+') as f_out:
        for i in range(10):
            if function_name == "delete":
                testacc_clean, testacc, testaccnew1, testaccnew2, testaccnew3 = baseline_delete()
            elif function_name == "add":
                testacc_clean, testacc, testaccnew1, testaccnew2, testaccnew3 = baseline_add()
            # testacc = 1.01
            # testaccnew1 = 1.01
            # testaccnew2 = 1.01
            # testaccnew3 = 1.01
            f_out.write(str(testacc_clean) + ' '+ str(testacc)+ ' '+str(testaccnew1)+ ' '+str(testaccnew2)+ ' '+str(testaccnew3)+"\n")

    # print("The original base model")
    #trained_dis_base(adj_norm, adj_label, if_ori = True)  #
    # print("The modified model base model")
    #trained_dis_base(adj_norm, new_adj, if_ori=False)
    #print("The modified model base model using x_tilde")
    #trained_dis_base(adj_norm, x_tilde_out, if_ori=False)
    # print("finish")

    # baseline()
    # base_line2()
    # base_line3()
