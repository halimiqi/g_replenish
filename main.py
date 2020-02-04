import tensorflow as tf
import random
import tensorflow.contrib.slim as slim
#from utils import mkdir_p
from utils import randomly_add_edges, randomly_delete_edges, randomly_flip_features,flip_features_fix_attr
from utils import add_edges_between_labels, denoise_ratio, get_noised_indexes
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
seed = 152   # last random seed is 141           0.703
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
#flags.DEFINE_integer('n_class', 6, 'Number of epochs to train.')
##### this is for gae part
flags.DEFINE_integer('n_clusters', 7, 'Number of epochs to train.')    # this one can be calculated according to labels
flags.DEFINE_string("target_index_list","10,35", "The index for the target_index")
flags.DEFINE_integer('epochs', 700, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('hidden3', 32, 'Number of units in graphite hidden layers.')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
####### for clean gcn training and test
flags.DEFINE_float('gcn_learning_rate', 0.01, 'Initial learning rate.')
#flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('gcn_hidden1', 16, 'Number of units in hidden layer 1.')
#flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('gcn_weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for: early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
###########################
flags.DEFINE_float('dropout', 0.3, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('g_scale_factor', 1- 0.75/2, 'the parametor for generate fake loss')
flags.DEFINE_float('d_scale_factor', 0.25, 'the parametor for discriminator real loss')
flags.DEFINE_float('g_gamma', 1e-06, 'the parametor for generate loss, it has one term with encoder\'s loss')
flags.DEFINE_float('G_KL_r', 0.1, 'The r parameters for the G KL loss')
flags.DEFINE_float('mincut_r', 0.3, 'The r parameters for the cutmin loss orth loss')
flags.DEFINE_float('autoregressive_scalar', 0.2, 'the parametor for graphite generator')
flags.DEFINE_string('model', 'gae_gan', 'Model string.')
flags.DEFINE_string('generator', 'dense', 'Which generator will be used') # the options are "inner_product", "graphite", "graphite_attention", "dense_attention" , "dense"
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')
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
flags.DEFINE_integer("k", 1000, "The k edges to delete")
flags.DEFINE_integer("k_features", 200, "The k nodes to flip features")
flags.DEFINE_float('ratio_loss_fea', 0, 'the ratio of generate loss for features')
flags.DEFINE_integer('delete_edge_times', 1, 'sample times for delete K edges. We use this to average the x_tilde(normalized adj) got from generator')
flags.DEFINE_integer('baseline_target_budget', 5, 'the parametor for graphite generator')
flags.DEFINE_integer("op", 1, "Training or Test")
flags.DEFINE_float("reward_para", "2.0", "The hyper parameters for reward")
###############################
if_drop_edge = True
if_save_model = False
# if train the discriminator
if_train_dis = True
restore_trained_our = False
showed_target_idx = 0   # the target index group of targets you want to show
run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
###################################
### read and process the graph
model_str = FLAGS.model
dataset_str = FLAGS.dataset
# Load data
# _A_obs, _X_obs, _z_obs = utils.load_npz('data/citeseer.npz')
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)
# _A_obs = _A_obs + _A_obs.T #变GCN_ori as GCN
# _A_obs[_A_obs > 1] = 1
# adj = _A_obs

adj_norm, adj_norm_sparse = preprocess_graph(adj)

#_K = _z_obs.max()+1 #类别个数
_K = y_train.shape[1]
features_normlize = normalize(features.tocsr(), axis=0, norm='max')
features = sp.csr_matrix(features_normlize)

# adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
# adj = adj_train
if FLAGS.features == 0:
    features = sp.identity(features.shape[0])  # featureless
# Some preprocessing

#placeholders = {
#    'features': tf.sparse_placeholder(tf.float32, name="ph_features"),
#    'adj': tf.sparse_placeholder(tf.float32, name="ph_adj"),
#    'adj_orig': tf.sparse_placeholder(tf.float32, name="ph_orig"),
#    'dropout': tf.placeholder_with_default(0., shape=(), name="ph_dropout"),
    # 'node_labels': tf.placeholder(tf.float32, name="ph_node_labels"),
    # 'node_ids': tf.placeholder(tf.float32, name="ph_node_ids")
#}

num_nodes = adj.shape[0]
features_csr = features
features_csr = features_csr.astype('float32')
features = sparse_to_tuple(features.tocoo())
num_features = features[2][1]
features_nonzero = features[1].shape[0]
n_class = _K

gpu_id = 1

# Create model

    #session part
cost_val = []
acc_val = []
cost_val = []
acc_val = []
val_roc_score = []

def get_new_adj(feed_dict, sess, model):
    new_adj = model.new_adj.eval(session=sess, feed_dict=feed_dict)
    new_adj = new_adj - np.diag(np.diagonal(new_adj))
    return new_adj

def get_new_feature(feed_dict, sess,flip_features_csr, feature_entry, model):
    new_indexes = model.flip_feature_indexes.eval(session = sess, feed_dict = feed_dict)
    flip_features_lil = flip_features_csr.tolil()
    for index in new_indexes:
        for j in feature_entry:
            flip_features_lil[index, j] = 1 - flip_features_lil[index, j]
    return flip_features_lil.tocsr()
# Train model
def train():
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]),
                                        shape=adj_orig.shape)  # delete self loop
    adj_orig.eliminate_zeros()
    #adj_new = randomly_add_edges(adj_orig, k=FLAGS.k)  # randomly add new edges
    adj_new , add_idxes= add_edges_between_labels(adj_orig, FLAGS.k*2, y_train)
    #features_new_csr = randomly_flip_features(features_csr, k = FLAGS.k, seed = seed+5) # randomly add new features
    fixed_entry = list(np.arange(100))
    features_new_csr = flip_features_fix_attr(features_csr, k = FLAGS.k_features * 2, seed = seed + 5, fixed_list = fixed_entry)
    feature_new = sparse_to_tuple(features_new_csr.tocoo())
    ####################   check the laplacian lower bound ##########
    row_sum = adj_new.sum(1).A1
    row_sum = sp.diags(row_sum)
    L = row_sum - adj_new
    ori_Lap = features_new_csr.transpose().dot(L).dot(features_new_csr)
    ori_Lap_trace = ori_Lap.diagonal().sum()
    ori_Lap_log = np.log(ori_Lap_trace)

    ###################   check the laplacian upper bound #############
    row_sum = adj_orig.sum(1).A1
    row_sum = sp.diags(row_sum)
    L = row_sum - adj_orig
    clean_Lap = features_new_csr.transpose().dot(L).dot(features_new_csr)
    clean_Lap_trace = clean_Lap.diagonal().sum()
    clean_Lap_log = np.log(clean_Lap_trace)
    ####################### the clean and noised GCN  ############################
    testacc_clean, valid_acc_clean = GCN.run(FLAGS.dataset, adj_orig, features_csr,y_train,y_val, y_test, train_mask, val_mask, test_mask, name = "clean")
    testacc, valid_acc = GCN.run(FLAGS.dataset, adj_new,features_new_csr, y_train,y_val, y_test, train_mask, val_mask, test_mask, name = "original")
    testacc_upper, valid_acc_upper = GCN.run(FLAGS.dataset, adj_new, features_csr,y_train,y_val, y_test, train_mask, val_mask, test_mask, name="upper_bound")
    ###########
    print(testacc_clean)
    print(testacc)
    print(testacc_upper)
    ###########
    ##############################################################################
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
        'clean_mask': tf.placeholder(tf.int32),
        'noised_mask': tf.placeholder(tf.int32),
        'noised_num':tf.placeholder(tf.int32)
        # 'node_labels': tf.placeholder(tf.float32, name = "ph_node_labels"),
        # 'node_ids' : tf.placeholder(tf.float32, name = "ph_node_ids")
    }
    # build models
    model = None
    adj_clean = adj_orig.tocoo()
    adj_clean_tensor = tf.SparseTensor(indices =np.stack([adj_clean.row,adj_clean.col], axis = -1),
                                       values = adj_clean.data, dense_shape = adj_clean.shape )
    if model_str == "gae_gan":
        model = gaegan(placeholders, num_features, num_nodes, features_nonzero,
                       new_learning_rate, indexes_add = add_idxes,
                       adj_clean = adj_clean_tensor)
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
                                  new_learning_rate = new_learning_rate,
                                  ori_reg_log = ori_Lap_log,
                                  placeholders = placeholders
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
    ### initial clean and noised_mask
    clean_mask = np.array([1,2,3,4,5])
    noised_mask = np.array([6,7,8,9,10])
    noised_num = noised_mask.shape[0] / 2
    ##################################
    feed_dict = construct_feed_dict(adj_norm, adj_label, feature_new,clean_mask, noised_mask,noised_num,  placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    # pred_dis_res = model.vaeD_tilde.eval(session=sess, feed_dict=feed_dict)
    ### debug #####
    #testnoised_index, testnew_indexes, test_sampled_dist = sess.run([model.test_noised_index,
    #                                             model.test_new_indexes,
    #                                              model.test_sampled_dist],
    #                                             feed_dict = feed_dict)
    #############

    #### save new_adj without norm#############
    if restore_trained_our:
        modified_adj = get_new_adj(feed_dict,sess, model)
        modified_adj = sp.csr_matrix(modified_adj)
        sp.save_npz("transfer_new/transfer_1216_1/qq_5000_gaegan_new.npz", modified_adj)
        sp.save_npz("transfer_new/transfer_1216_1/qq_5000_gaegan_ori.npz", adj_new)
        print("save the loaded adj")
    # print("before training generator")
    #####################################################
    ##  get all variables in the model
    def model_summary():
        model_vars = tf.trainable_variables()
        slim.model_analyzer.analyze_vars(model_vars,print_info=True)
    model_summary()
    #####################################################
    reg_trace = 0
    edge_percent = 0
    last_reg = 0
    current_reg = 0
    G_loss_min = 1000
    inter_num = 0
    D_loss = 0
    density_0 = 0
    inter_num_0 = 0
    for epoch in range(FLAGS.epochs):
        t = time.time()
        # run Encoder's optimizer
        #sess.run(opt.encoder_min_op, feed_dict=feed_dict)
        # run G optimizer  on trained mode
        ########
        last_reg = current_reg
        if restore_trained_our:
            # _, current_reg, reg_trace, edge_percent = sess.run([opt.G_min_op, opt.reg_log, opt.reg_trace,
            #                                                     opt.percentage_edge], feed_dict=feed_dict, options=run_options)
            _, current_reg, edge_percent = sess.run([opt.G_min_op, opt.reg,
                                                                opt.percentage_edge], feed_dict=feed_dict,
                                                               options=run_options)
            sess.run(tf.assign(opt.last_reg, current_reg))
        else: # it is the new model
            if epoch > int(FLAGS.epochs / 2):  ## here we can contorl the manner of new model
                _= sess.run([opt.G_min_op], feed_dict=feed_dict,
                                                           options=run_options)

                new_adj = get_new_adj(feed_dict,sess, model)
                # sess.run(tf.assign(opt.last_reg, current_reg))
            else:
                _, x_tilde = sess.run([opt.D_min_op, model.realD_tilde], feed_dict = feed_dict, options=run_options)
                #### get the noised
                if epoch == int(FLAGS.epochs / 2):
                    noised_indexes, clean_indexes =  get_noised_indexes(x_tilde, adj_new)
                ####
                    feed_dict.update({placeholders["noised_mask"]: noised_indexes})
                    feed_dict.update({placeholders["clean_mask"]: clean_indexes})
                    feed_dict.update({placeholders["noised_num"]: len(noised_indexes)/2}) # because we noly sample one side
                #sess.run(tf.assgin(opt.noised_indexes, noised_indexes, validate_shape = False))
                #_,current_reg, edge_percent, density_ori, inter_num = sess.run([opt.D_min_op,opt.reward_and_per,
                #                                                                opt.percentage_edge,
                #                                                                model.vaeD_density,
                #                                                                model.inter_num],
                #                                                               feed_dict = feed_dict,
                #                                                               options = run_options)
        ##
        #if epoch == 0:
            #density_0 = density_ori
            # inter_num_0 = inter_num
        ##
        if epoch % 50 == 0:
            if epoch > int(FLAGS.epochs / 2):
                print("This is the vae part")
            else:
                print("This is training the discriminator")
            print("Epoch:", '%04d' % (epoch + 1),
                  "time=", "{:.5f}".format(time.time() - t))
            G_loss,D_loss, new_learn_rate_value = sess.run([opt.G_comm_loss,opt.D_loss,new_learning_rate],feed_dict=feed_dict,  options = run_options)
            #new_adj = get_new_adj(feed_dict, sess, model)
            #new_adj = model.new_adj_output.eval(session = sess, feed_dict = feed_dict)
            #temp_pred = new_adj.reshape(-1)
            #temp_ori = adj_norm_sparse.todense().A.reshape(-1)
            #temp_ori = adj_label_sparse.todense().A.reshape(-1)
            #mutual_info = normalized_mutual_info_score(temp_pred, temp_ori)
            #print("Step: %d,G: loss=%.7f ,D: loss= %.7f,info_score = %.6f, LR=%.7f" % (epoch, G_loss,D_loss, mutual_info,new_learn_rate_value))
            print("Step: %d,G: loss=%.7f ,D: loss= %.7f, LR=%.7f" % (epoch, G_loss,D_loss, new_learn_rate_value))
            ## here is the debug part of the model#################################

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
    new_adj_norm, new_adj_norm_sparse = preprocess_graph(new_adj)
    new_adj_norm_sparse_csr = new_adj_norm_sparse.tocsr()
    ##### get new features ###########
    new_features_csr = get_new_feature(feed_dict, sess, features_new_csr, fixed_entry, model)
    ##################################
    # modified_model = GCN.GCN(sizes, new_adj_norm_sparse_csr, features_csr, with_relu=True, name="surrogate", gpu_id=gpu_id)
    # modified_model.train(new_adj_norm_sparse_csr, split_train, split_val, node_labels)
    # modified_acc = modified_model.test(split_unlabeled, node_labels, new_adj_norm_sparse_csr)
    testacc_new_adj, valid_acc_new_adj = GCN.run(FLAGS.dataset,new_adj_sparse,features_new_csr,y_train,y_val, y_test, train_mask, val_mask, test_mask, name = "modified")
    testacc_new_adj_cleanfea, valid_acc_new_adj_cleanfea = GCN.run(FLAGS.dataset,new_adj_sparse,features_csr,y_train,y_val, y_test, train_mask, val_mask, test_mask, name = "modified")
    testacc_new_adj_fea, valid_acc_new_adj_fea = GCN.run(FLAGS.dataset, new_adj_sparse, new_features_csr, y_train, y_val, y_test,
                                         train_mask, val_mask, test_mask, name="modified")
    new_adj = get_new_adj(feed_dict, sess, model)
    new_adj = new_adj - np.diag(np.diag(new_adj))
    new_adj_sparse = sp.csr_matrix(new_adj)
    #testacc_new2, valid_acc_new = GCN.run(FLAGS.dataset,adj_new,new_features_csr,y_train,y_val, y_test, train_mask, val_mask, test_mask,  name="modified2")
    #np.save("./data/hinton/hinton_new_adj_48_0815.npy", new_adj)
    #roc_score, ap_score = get_roc_score(test_edges, test_edges_false,feed_dict, sess, model)
    ##### The final results ####
    print("*" * 30)
    print("the final results:\n")
    print("*" * 30)
    print("The clean acc is: ")
    print(testacc_clean)
    print("*#" * 15)
    print("The noised adj and features acc is: ")
    print(testacc)
    print("*#"* 15)
    print("The noisd adj only acc is: ")
    print(testacc_upper)
    print("*#" * 15)
    print("The only modify adj acc is(with noised feature) : ")
    print(testacc_new_adj)
    print("*#" * 15)
    print("The only modify adj acc is(with clean feature):")
    print(testacc_new_adj_cleanfea)
    # print("The only modify feature acc is : ")
    # print(testacc_new2)
    # print("*#" * 15)
    print("The modify both adj and feature and acc is(noised both) : ")
    print(testacc_new_adj_fea)
    return new_adj,testacc_clean, testacc_upper, testacc_new_adj, testacc_new_adj_fea    #, testacc_new2, testacc_new3

FLAGS = flags.FLAGS
if __name__ == "__main__":
    #train_dis_base()
    current_time = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    with open("results/results_%d_%s.txt"%(FLAGS.k, current_time), 'w+') as f_out:
        f_out.write("clean_acc" +" "+ "noisedadj_acc" + ' ' + 'modify_adj'+ ' ' + 'modify_both' + ' ' + 'modify_both' + "\n")
        for i in range(1):
            new_adj,testacc_clean,testacc_noised_adj,  testacc_adj, testaccnew_adjfea = train()
            # testacc = 1.01
            # testaccnew1 = 1.01
            # testaccnew2 = 1.01
            # testaccnew3 = 1.01
            f_out.write(str(testacc_clean)+" "+str(testacc_noised_adj)+ ' '+str(testaccnew_adjfea)+ ' '+"\n")
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
