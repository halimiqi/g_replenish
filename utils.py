# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
import scipy.sparse as sp
import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy import linalg
import random
import math
#from numba import jit


def load_npz_edges(file_name):
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    dict_of_lists = {}
    with np.load(file_name) as loader:
        loader = dict(loader)
        num_nodes = loader['adj_shape'][0]
        indices = loader['adj_indices']
        indptr = loader['adj_indptr']
        for i in range(num_nodes):
            if len(indices[indptr[i]:indptr[i+1]]) > 0:
                dict_of_lists[i] = indices[indptr[i]:indptr[i+1]].tolist()

    return dict_of_lists


def load_npz(file_name):
    """Load a SparseGraph from a Numpy binary file.

    Parameters
    ----------
    file_name : str
        Name of the file to load.

    Returns
    -------
    sparse_graph : gust.SparseGraph
        Graph in sparse matrix format.

    """
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    with np.load(file_name,allow_pickle=True) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                              loader['adj_indptr']), shape=loader['adj_shape'])
        #import pdb; pdb.set_trace()
        if 'attr_data' in loader:
            attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                                   loader['attr_indptr']), shape=loader['attr_shape'])
        else:
            attr_matrix = None

        labels = loader.get('labels')

    return adj_matrix, attr_matrix, labels


def largest_connected_components(adj, n_components=1):
    """Select the largest connected components in the graph.返回最大连通子图

    Parameters
    ----------
    sparse_graph : gust.SparseGraph
        Input graph.
    n_components : int, default 1
        Number of largest connected components to keep.

    Returns
    -------
    sparse_graph : gust.SparseGraph
        Subgraph of the input graph where only the nodes in largest n_components are kept.

    """
    _, component_indices = connected_components ( adj ) #Return the length-N array of each node's label in the connected components.
    component_sizes = np.bincount(component_indices) #Count number of occurrences of each value in array of non-negative ints.
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending 最大连通子图中的节点存成list
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep


    ]
    print("Selecting {0} largest connected components".format(n_components))
    return nodes_to_keep


def train_val_test_split_tabular(*arrays, train_size=0.5, val_size=0.3, test_size=0.2, stratify=None, random_state=None):
    """
    Split the arrays or matrices into random train, validation and test subsets.
    #train_test_split重写成可以分出验证集的函数

    Parameters
    ----------
    *arrays : sequence of indexables with same length / shape[0]
            Allowed inputs are lists, numpy arrays or scipy-sparse matrices.
    train_size : float, default 0.5
        Proportion of the dataset included in the train split.
    val_size : float, default 0.3
        Proportion of the dataset included in the validation split.
    test_size : float, default 0.2
        Proportion of the dataset included in the test split.
    stratify : array-like or None, default None
        If not None, data is split in a stratified fashion, using this as the class labels. #是否为标签均衡的数据
    random_state : int or None, default None
        Random_state is the seed used by the random number generator;

    Returns
    -------
    splitting : list, length=3 * len(arrays)
        List containing train-validation-test split of inputs.

    """
    if len(set(array.shape[0] for array in arrays)) != 1:
        raise ValueError("Arrays must have equal first dimension.")
    idx = np.arange(arrays[0].shape[0])
    idx_train_and_val, idx_test = train_test_split(idx,
                                                   random_state=random_state,
                                                   train_size=(train_size + val_size),
                                                   test_size=test_size,
                                                   stratify=stratify)
    if stratify is not None:
        stratify = stratify[idx_train_and_val]
    idx_train, idx_val = train_test_split(idx_train_and_val,
                                          random_state=random_state,
                                          train_size=(train_size / (train_size + val_size)),
                                          test_size=(val_size / (train_size + val_size)),
                                          stratify=stratify)
    result = []
    for X in arrays:
        result.append(X[idx_train])
        result.append(X[idx_val])
        result.append(X[idx_test])
    return result

def preprocess_graph(adj):
    adj_ = adj + sp.eye(adj.shape[0]) #对应加单位阵的操作
    rowsum = adj_.sum(1).A1 #对应求sum_{j}(A_{ij})
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5)) #对应求D^(-1/2)
    #degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -1)) #对应求D^(-1/2)
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).T.dot(degree_mat_inv_sqrt).tocsr() #求A^hat并将其转化为Compressed Sparse Row format
    #import pdb; pdb.set_trace()
    #adj_normalized = adj_
    return adj_normalized

'''
def cal_scores(A):
    eig_vals, eig_vecl = linalg.eig(A.todense(), left = True, right = False)
    eig_vals, eig_vecr = linalg.eig(A.todense(), left = False, right = True)
    eig_idx = eig_vals.argmax()
    eig_l = eig_vecl[eig_idx]
    eig_r = eig_vecr[eig_idx]
    eig_l[eig_l < 0 ] = -eig_l[eig_l < 0 ]
    eig_r[eig_r < 0 ] = -eig_r[eig_r < 0 ]
    print ("The largest eigenvalue of A is {}".format(eig_vals.max()))
    scores = (eig_l.reshape(eig_l.shape[0],1) * eig_r).flatten()
    return scores
'''

'''
def cal_scores(A):
    eig_vals, eig_vec = linalg.eigh(A.todense())
    eig_idx = eig_vals.argmax()
    eig = eig_vec[eig_idx]
    eig[eig < 0 ] = -eig[eig < 0 ]
    print ("The largest eigenvalue of A is {}".format(eig_vals.max()))
    scores = (eig.reshape(eig.shape[0],1) * eig).flatten()
    #import pdb; pdb.set_trace()
    return scores
'''


def cal_scores(A, X):
    #eig_vals, eig_vec = linalg.eigh(preprocess_graph(A).dot(X).dot(X.T).todense())
    eig_vals, eig_vec = linalg.eigh(preprocess_graph(A).todense())
    #eig_vals, eig_vec = linalg.eigh(A.todense())
    eig_idx = eig_vals.argmax()
    eig = eig_vec[eig_idx]
    eig[eig < 0 ] = -eig[eig < 0 ]
    print ("The largest eigenvalue of A is {}".format(eig_vals.max()))
    #scores = (eig.reshape(eig.shape[0],1) * eig).flatten()
    scores = (eig.reshape(eig.shape[0],1) * eig)
    print ("The largest score of perturbation on A is {}".format(scores.max()))
    #import pdb; pdb.set_trace()
    #return scores.flatten()
    return scores

# the mat version of uAu uDu
def cal_scores_mat_AD(A, X, filtered_edges,r=1, k="half", lambda_method = "sum", with_X = True, X_max_part = True):
    results = []
    #A_add_I = A + sp.eye(A.shape[0])  # 对应加单位阵的操作
    rowsum = A.sum(1).A1  # 对应求sum_{j}(A_{ij})
    # degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5)) #对应求D^(-1/2)
    degree_mat = sp.diags(rowsum)  # 对应求D^(-1/2)
    D_min = rowsum.min()
    eig_vals, eig_vec = linalg.eigh(A.todense(), degree_mat.todense())
    # define the k
    abs_V = len(eig_vals)
    tmp_k = 0
    if k == "fix":
        tmp_k = 128
    else:
        tmp_k = int(abs_V / 2)
    for j in range(len(filtered_edges)):
        filtered_edge = filtered_edges[j]
        delta_A = sp.lil_matrix((A.shape[0],A.shape[1]))
        delta_A[filtered_edge[0], filtered_edge[1]] = 1 - 2*A[filtered_edge[0], filtered_edge[1]]
        delta_A[filtered_edge[1], filtered_edge[0]] = 1 - 2*A[filtered_edge[0], filtered_edge[1]]   # because it is unweighted
        delta_rowsum = delta_A.sum(1).A1
        delta_D = sp.diags(delta_rowsum)
        eig_vals_res = np.zeros(len(eig_vals))
        #eig_vec_trans = eig_vec.T
        uAu_mat  = np.asarray(np.dot(eig_vec.T, delta_A.todense()).dot(eig_vec))
        uDu_mat = np.asarray(np.dot(eig_vec.T, delta_D.todense()).dot(eig_vec))
        uAu = uAu_mat.diagonal()
        uDu = uDu_mat.diagonal()
        eig_vals_res = eig_vals+ (uAu - eig_vals* uDu)
        if lambda_method == "sum":
            if r==1:
                eig_vals_res =np.abs(eig_vals_res * (1/D_min))
            else:
                for itr in range(1,r):
                    eig_vals_res = eig_vals_res + np.power(eig_vals_res, itr+1)
                eig_vals_res = np.abs(eig_vals_res) * (1 / D_min)
        else:
            eig_vals_res = np.square((eig_vals_res + np.ones(len(eig_vals_res))))
        #initial the final loss
        final_loss = 0
        if not with_X:
            eig_vals_res = np.square(eig_vals_res)
            eig_vals_res = np.sort(eig_vals_res)
            final_loss = np.sum(eig_vals_res[0:(abs_V - tmp_k)])
            final_loss = pow(final_loss, 0.5)
            if lambda_method == "sum":
                final_loss = final_loss *(rowsum.sum() + 2 * (1 - 2*A[filtered_edge[0], filtered_edge[1]]))
        else:
            if X_max_part:
                eig_vals_argk = np.argpartition(eig_vals_res, -tmp_k)[-tmp_k:]
                eig_vals_k = eig_vals_res[eig_vals_argk]
                u_k = eig_vec[:, eig_vals_argk]
                eig_vals_matk = np.diag(eig_vals_k)
                k_f = eig_vals_matk.dot(u_k.T).dot(X.todense())
                k_1 = np.mean(k_f, axis=1)
                final_loss = np.sum(k_1)
            else:
                # eig_vals_argk = np.argpartition(eig_vals_res, -tmp_k)[-tmp_k:]
                # eig_vals_k = eig_vals_res[eig_vals_argk]
                # u_k = eig_vals_res[:, eig_vals_k]
                eig_vals_mat = np.diag(eig_vals_res)
                n_f = eig_vals_mat.dot(eig_vec.T).dot(X.todense())
                n_1 = np.array(np.mean(n_f, axis=1))
                final_loss = n_1[filtered_edge[1], 0]

        results.append(final_loss)
        #return final_loss
        print("The mat_version progress:%f%%" % (((j + 1) / (len(filtered_edges))) * 100), end='\r', flush=True)
    print("\n")
    return np.array(results)

def cal_scores_oneedge_AD(A, X, filtered_edges, r=1, k="half", lambda_method = "sum", with_X = True, X_max_part = True):
    results = []
    #A_add_I = A + sp.eye(A.shape[0])  # 对应加单位阵的操作
    rowsum = A.sum(1).A1  # 对应求sum_{j}(A_{ij})
    # degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5)) #对应求D^(-1/2)
    degree_mat = sp.diags(rowsum)  # 对应求D^(-1/2)
    D_min = rowsum.min()
    eig_vals, eig_vec = linalg.eigh(A.todense(), degree_mat.todense())
    # define the k
    abs_V = len(eig_vals)
    tmp_k = 0
    if k == "fix":
        tmp_k = 128
        #tmp_k = 2
    else:
        tmp_k = int(abs_V / 2)
    for j in range(len(filtered_edges)):
        filtered_edge = filtered_edges[j]
        eig_vals_res = np.zeros(len(eig_vals))
        eig_vals_res = (1 - 2*A[filtered_edge[0], filtered_edge[1]]) * (2* eig_vec[filtered_edge[0],:] * eig_vec[filtered_edge[1],:] - eig_vals *
                                                                        ( np.square(eig_vec[filtered_edge[0],:]) + np.square(eig_vec[filtered_edge[1],:])))
        eig_vals_res = eig_vals + eig_vals_res

        if lambda_method == "sum":
            if r==1:
                eig_vals_res =np.abs(eig_vals_res * (1/D_min))
            else:
                # eig_vals_res = np.abs(eig_vals_res)
                for itr in range(1,r):
                    eig_vals_res = eig_vals_res + np.power(eig_vals_res, itr+1)
                eig_vals_res = np.abs(eig_vals_res) * (1/D_min)
        else:
            eig_vals_res = np.square((eig_vals_res + np.ones(len(eig_vals_res))))
        #initial the final loss
        final_loss = 0
        if not with_X:
            eig_vals_res = np.square(eig_vals_res)
            eig_vals_res = np.sort(eig_vals_res)
            needed = eig_vals_res[0:(abs_V - tmp_k)]
            final_loss = np.sum(eig_vals_res[0:(abs_V - tmp_k)])
            final_loss = pow(final_loss, 0.5)
            if lambda_method == "sum":
                final_loss = final_loss * (rowsum.sum() + 2 * (1 - 2 * A[filtered_edge[0], filtered_edge[1]]))
        else:
            if X_max_part:
                eig_vals_argk = np.argpartition(eig_vals_res, -tmp_k)[-tmp_k:]
                eig_vals_k = eig_vals_res[eig_vals_argk]
                u_k = eig_vec[:,eig_vals_argk]
                eig_vals_matk = np.diag(eig_vals_k)
                k_f = eig_vals_matk.dot(u_k.T).dot(X.todense())
                k_1 = np.mean(k_f, axis=1)
                final_loss = np.sum(k_1)
            else:
                # eig_vals_argk = np.argpartition(eig_vals_res, -tmp_k)[-tmp_k:]
                # eig_vals_k = eig_vals_res[eig_vals_argk]
                #u_k = eig_vals_res[:, eig_vals_k]
                eig_vals_mat = np.diag(eig_vals_res)
                n_f = eig_vals_mat.dot(eig_vec.T).dot(X.todense())
                n_1 = np.array(np.mean(n_f, axis=1))
                final_loss = n_1[filtered_edge[1],0]

        results.append(final_loss)
        print("The one_edge_version progress:%f%%" % (((j + 1) / (len(filtered_edges))) * 100), end='\r', flush=True)
    print("\n")
    return np.array(results)

#@jit(nopython=True)
def cal_scores_oneedge_AD_uXu(A, X, X_mean,eig_vals, eig_vec, filtered_edges, r=1, k="fix", lambda_method = "nosum", with_X = True, X_max_part = True):
    results = []
    #A_processed = preprocess_graph(A).tolil()
    #edge_ixs = np.array(A.nonzero()).T
    A = A + sp.eye(A.shape[0])  # 对应加单位阵的?
    A[A>1] = 1
    rowsum = A.sum(1).A1  # 对应求sum_{j}(A_{ij})
    # degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5)) #对应求D^(-1/2)
    #degree_mat = sp.diags(rowsum)  # 对应求D^(-1/2)
    D_min = rowsum.min()
    #eig_vals, eig_vec = linalg.eigh(A.todense(), degree_mat.todense())
    # define the k
    abs_V = len(eig_vals)
    tmp_k = 0
    if k == "fix":
        tmp_k = 128
        #tmp_k = 2
    else:
        tmp_k = int(abs_V / 2)


    return_values = []
    # X_mean = np.sum(X, axis = 1)
    #X_mean = np.ones(A.shape[0])

    for j in range(len(filtered_edges)):
        filtered_edge = filtered_edges[j]
        eig_vals_res = np.zeros(len(eig_vals))
        eig_vals_res = (1 - 2*A[filtered_edge[0], filtered_edge[1]]) * (2* eig_vec[filtered_edge[0],:] * eig_vec[filtered_edge[1],:] - eig_vals *
                                                                        ( np.square(eig_vec[filtered_edge[0],:]) + np.square(eig_vec[filtered_edge[1],:])))
        eig_vals_res = eig_vals + eig_vals_res

        if lambda_method == "sum":
            if r==1:
                eig_vals_res =np.abs(eig_vals_res * (1/D_min))
            else:
                # eig_vals_res = np.abs(eig_vals_res)
                for itr in range(1,r):
                    eig_vals_res = eig_vals_res + np.power(eig_vals_res, itr+1)
                eig_vals_res = np.abs(eig_vals_res) * (1/D_min)
        else:
            eig_vals_res = np.square((eig_vals_res + np.ones(len(eig_vals_res))))
            eig_vals_res = np.power(eig_vals_res, r)
            #eig_vals_res = np.square(eig_vals_res)

        eig_vals_idx = np.argsort(eig_vals_res)  # from small to large
        #eig_vals_k = eig_vals_res[eig_vals_argk]
        # Original_0
        # u_k = eig_vec[:,eig_vals_idx[:tmp_k]]
        # u_x_mean = u_k.T.dot(X_mean)
        # return_values.append(np.sum(np.abs(u_x_mean)))
        # Original_1
        eig_vals_k_sum = eig_vals_res[eig_vals_idx[:tmp_k]].sum()
        u_k = eig_vec[:,eig_vals_idx[:tmp_k]]
        u_x_mean = u_k.T.dot(X_mean)
        return_values.append(eig_vals_k_sum * np.sum(np.abs(u_x_mean)))

        print("The one_edge_version progress:%f%%" % (((j + 1) / (len(filtered_edges))) * 100), end='\r', flush=True)
    #ixs_arr = np.array(return_ixs)
    #lambda_hat_uv = sp.coo_matrix((return_values, (ixs_arr[:, 0], ixs_arr[:, 1])), shape=[len(filtered_edges), A.shape[0]])
    #struct_scores = np.mean(lambda_hat_uv.dot(X), axis=1)
    print("\n")

    return np.array(return_values)

def cal_scores_oneedge_onlyA(A, X, filtered_edges, r=1,k="half", lambda_method = "sum", with_XX = False):
    results = []
    #A_add_I = A + sp.eye(A.shape[0])  # 对应加单位阵的操作
    if not with_XX:
        rowsum = A.sum(1).A1  # 对应求sum_{j}(A_{ij})
        degree_mat = sp.diags(rowsum)  # 对应求D^(-1/2)
        D_min = rowsum.min()
        eig_vals, eig_vec = linalg.eigh(A.todense())

        # define the k
        abs_V = len(eig_vals)
        tmp_k = 0
        if k == "fix":
            tmp_k = 128
        else:
            tmp_k = int(abs_V / 2)
        uu_vec = np.dot(eig_vec.T, eig_vec).diagonal()
        for j in range(len(filtered_edges)):
            filtered_edge = filtered_edges[j]
            eig_vals_res = np.zeros(len(eig_vals))
            eig_vals_res = (1 - 2*A[filtered_edge[0], filtered_edge[1]]) * (2* eig_vec[filtered_edge[0],:] * eig_vec[filtered_edge[1],:])
            eig_vals_res = eig_vals + np.divide(eig_vals_res, uu_vec) # if there is only A, then the uu must be divided
            if lambda_method == "sum":
                if r==1:
                    eig_vals_res =np.abs(eig_vals_res * (1/D_min))
                else:
                    for itr in range(1,r):
                        eig_vals_res = eig_vals_res + np.power(eig_vals_res, itr+1)
                    eig_vals_res = np.abs(eig_vals_res) * (1 / D_min)
            else:
                eig_vals_res = np.square((eig_vals_res + np.ones(len(eig_vals_res))))
            #initial the final loss
            final_loss = 0
            eig_vals_res = np.square(eig_vals_res)
            eig_vals_res = np.sort(eig_vals_res)
            needed = eig_vals_res[0:(abs_V - tmp_k)]
            final_loss = np.sum(eig_vals_res[0:(abs_V - tmp_k)])
            final_loss = pow(final_loss, 0.5)
            if lambda_method == "sum":
                final_loss = final_loss * (rowsum.sum() + 2 * (1 - 2 * A[filtered_edge[0], filtered_edge[1]]))
            results.append(final_loss)
    else:
        xx_mat = X.dot(X.T)
        ashape = A.todense()
        xx_shape = xx_mat.todense()
        eig_vals, eig_vec_l,eig_vec_r  = linalg.eig(A.todense(),xx_mat.todense(), left=True, right=True)
        # difine the K and V for sigma
        abs_V = len(eig_vals)
        tmp_k = 0
        if k == "fix":
            tmp_k = 128
        else:
            tmp_k = int(abs_V / 2)
        for j in range(len(filtered_edges)):
            filtered_edge = filtered_edges[j]
            eig_vals_res = (1 - 2 * A[filtered_edge[0], filtered_edge[1]]) * (
                            eig_vec_l[filtered_edge[0], :] * eig_vec_r[filtered_edge[1], :] + eig_vec_l[filtered_edge[1], :] * eig_vec_r[filtered_edge[0], :])  # The uAu
            uxxu = np.array(eig_vec_l.T.dot(xx_mat.todense()).dot(eig_vec_r).diagonal())[0,:]
            uxxu = uxxu[np.where(eig_vals.real != float("inf"))]
            eig_vals_res = eig_vals_res[np.where(eig_vals.real != float("inf"))]
            eig_vals = eig_vals[np.where(eig_vals.real != float("inf"))]
            eig_vals_res =eig_vals.real + np.divide(eig_vals_res, uxxu)
            if r==1:
                eig_vals_res =np.abs(eig_vals_res)
            else:
                for itr in range(1,r):
                    eig_vals_res = eig_vals_res + np.power(eig_vals_res, itr+1)
                eig_vals_res = np.abs(eig_vals_res)
            #calculate the final score
            final_loss = 0
            eig_vals_res = np.square(eig_vals_res)
            eig_vals_res = np.sort(eig_vals_res)
            if len(eig_vals_res)>=(abs_V- tmp_k):
                final_loss = np.sum(eig_vals_res[0:(abs_V - tmp_k)])
            else:
                final_loss = np.sum(eig_vals_res)
            final_loss = pow(final_loss, 0.5)
            results.append(final_loss)
        print("The one_edge_version_onlyA progress:%f%%" % (((j + 1) / (len(filtered_edges))) * 100), end='\r', flush=True)
    print("\n")
    return np.array(results)

def cal_scores_mat_onlyA(A, X, filtered_edges, r=1, k="half", lambda_method = "sum", with_XX = False):
    results = []
    if not with_XX:
        #A_add_I = A + sp.eye(A.shape[0])  # 对应加单位阵的操作
        rowsum = A.sum(1).A1  # 对应求sum_{j}(A_{ij})
        degree_mat = sp.diags(rowsum)  # 对应求D^(-1/2)
        D_min = rowsum.min()
        eig_vals, eig_vec = linalg.eigh(A.todense())
        # define the k
        abs_V = len(eig_vals)
        # the k can be changed
        tmp_k = 0
        if k == "fix":
            tmp_k = 128
        else:
            tmp_k = int(abs_V/2)
        uu = np.asarray(np.dot(eig_vec.T, eig_vec)).diagonal() # the transpose u multiply the u, it is actually u*v.
        for j in range(len(filtered_edges)):
            filtered_edge = filtered_edges[j]
            delta_A = sp.lil_matrix((A.shape[0],A.shape[1]))
            delta_A[filtered_edge[0], filtered_edge[1]] = 1 - 2*A[filtered_edge[0], filtered_edge[1]]
            delta_A[filtered_edge[1], filtered_edge[0]] = 1 - 2*A[filtered_edge[0], filtered_edge[1]]   # because it is unweighted
            eig_vals_res = np.zeros(len(eig_vals))
            uAu_mat  = np.asarray(np.dot(eig_vec.T, delta_A.todense()).dot(eig_vec))
            uAu = uAu_mat.diagonal()
            eig_vals_res = eig_vals+ np.divide(uAu, uu)
            # there are two choices here, r ==1 and r ==2
            if lambda_method == "sum":
                if r==1:
                    eig_vals_res =np.abs(eig_vals_res * (1/D_min))
                else:
                    for itr in range(1,r):
                        eig_vals_res = eig_vals_res + np.power(eig_vals_res, itr+1)
                    eig_vals_res = np.abs(eig_vals_res) * (1 / D_min)
            else:
                eig_vals_res = np.square((eig_vals_res + np.ones(len(eig_vals_res))))
            #initial the final loss
            final_loss = 0
            eig_vals_res = np.square(eig_vals_res)
            eig_vals_res = np.sort(eig_vals_res)
            final_loss = np.sum(eig_vals_res[0:(abs_V - tmp_k)])
            final_loss = pow(final_loss, 0.5)
            if lambda_method == "sum":
                final_loss = final_loss *(rowsum.sum() + 2 * (1 - 2*A[filtered_edge[0], filtered_edge[1]]))
            results.append(final_loss)
    else:
        xx_mat = X.dot(X.T)
        eig_vals, eig_vec_l, eig_vec_r = linalg.eig(A.todense(), xx_mat.todense(), left=True, right=True)
        abs_V = len(eig_vals)
        tmp_k = 0
        if k == "fix":
            tmp_k = 128
        else:
            tmp_k = int(abs_V / 2)
        for j in range(len(filtered_edges)):
            filtered_edge = filtered_edges[j]
            delta_A = sp.lil_matrix((A.shape[0],A.shape[1]))
            delta_A[filtered_edge[0], filtered_edge[1]] = 1 - 2*A[filtered_edge[0], filtered_edge[1]]
            delta_A[filtered_edge[1], filtered_edge[0]] = 1 - 2*A[filtered_edge[0], filtered_edge[1]]   # because it is unweighted
            eig_vals_res = np.zeros(len(eig_vals))
            uAu_mat  = np.asarray(np.dot(eig_vec_l.T, delta_A.todense()).dot(eig_vec_r))
            uAu = uAu_mat.diagonal()
            uxxu = np.array(eig_vec_l.T.dot(xx_mat.todense()).dot(eig_vec_r).diagonal())[0,:]
            # to delete the inf part
            uAu = uAu[np.where(eig_vals.real != float("inf"))]
            uxxu = uxxu[np.where(eig_vals.real != float("inf"))]
            eig_vals_res = eig_vals_res[np.where(eig_vals.real != float("inf"))]
            eig_vals = eig_vals[np.where(eig_vals.real != float("inf"))]
            eig_vals_res = eig_vals.real + np.divide(uAu, uxxu)
            if r == 1:
                eig_vals_res = np.abs(eig_vals_res)
            else:
                for itr in range(1, r):
                    eig_vals_res = eig_vals_res + np.power(eig_vals_res, itr + 1)
                eig_vals_res = np.abs(eig_vals_res)
            final_loss = 0
            eig_vals_res = np.square(eig_vals_res)
            eig_vals_res = np.sort(eig_vals_res)
            if len(eig_vals_res)>=(abs_V- tmp_k):
                final_loss = np.sum(eig_vals_res[0:(abs_V - tmp_k)])
            else:
                final_loss = np.sum(eig_vals_res)
            final_loss = pow(final_loss, 0.5)
            results.append(final_loss)
        #return final_loss

        print("The mat_version_onlyA progress:%f%%" % (((j + 1) / (len(filtered_edges))) * 100), end='\r', flush=True)
    print("\n")
    return np.array(results)
def cal_scores_controler(A,X,X_mean,eig_vals, eig_vec,filtered_edges,method = "oneedge",if_onlyA = False ,r=2, k="fix", lambda_method = "nosum", with_X = False,  X_max_part = False, with_XX = False):
    # A add I

    if method == "oneedge":
        A = A + sp.eye(A.shape[0])
        if if_onlyA == True:
            if with_X or X_max_part:
                print("There is no with_X method for only A\n")
            return cal_scores_oneedge_onlyA(A,X,filtered_edges,r,k, lambda_method=lambda_method,with_XX= with_XX)
        else:  # AD
            if with_XX:
                print("There is no with_XX method for only A")
            return cal_scores_oneedge_AD(A,X,filtered_edges,r,k, lambda_method, with_X,X_max_part)

    if method == "mat":
        A = A + sp.eye(A.shape[0])
        if if_onlyA:
            if with_X or X_max_part:
                print("There is no with_X method for only A\n")
            return cal_scores_mat_onlyA(A,X,filtered_edges,r,k, lambda_method = lambda_method, with_XX = with_XX)

        else:
            if with_XX:
                print("There is no with_XX method for only A")
            return cal_scores_mat_AD(A,X,filtered_edges,r,k, lambda_method, with_X,X_max_part)

    if method == "ulambdau":
        return cal_scores_oneedge_AD_uXu(A,X,X_mean,eig_vals,eig_vec,filtered_edges,r = r, k = k, lambda_method= lambda_method, with_X =False)

def k_edgedel(A, scores, dict_of_lists, k):
    N = A.shape[0]
    idxes = np.argsort(-scores)
    edge_pert = []
    for p in idxes:
        x = p // N
        y = p % N
        if x == y:
            continue
        if x not in dict_of_lists.keys() or y not in dict_of_lists.keys():
            continue
        if not x in dict_of_lists[y] or not y in dict_of_lists[x]:
            continue
        edge = np.array([x, y])
        if np.isin(edge[::-1], edge_pert).all():
           continue
        print ("The best edge for deletion is ({0} {1}), with score {2}".format(x, y, scores[p]))
        edge_pert.append(edge)
        #import pdb; pdb.set_trace()
        if len(edge_pert) >= k:
            break
    for edge in edge_pert:
        #import pdb; pdb.set_trace()
        A[tuple(edge)] = A[tuple(edge[::-1])] = 1 - A[tuple(edge)]
    adj = preprocess_graph(A)
    return adj, edge_pert

def randomly_add_edges(adj, k):
    num_nodes = adj.shape[0]
    adj_out = adj.copy()
    ## find the position which need to be add
    adj_orig_dense = adj.todense()
    flag_adj = np.triu(np.ones([num_nodes, num_nodes]), k=1) - np.triu(adj_orig_dense, k=1)
    idx_list = np.argwhere(flag_adj == 1)
    selected_idx_of_idx_list = np.random.choice(len(idx_list),size = k)
    selected_idx = idx_list[selected_idx_of_idx_list]
    adj_out[selected_idx[:,0],selected_idx[:,1]] = 1
    adj_out[selected_idx[:, 1], selected_idx[:, 0]] = 1
    return adj_out


def randomly_delete_edges(adj, k):
    num_nodes = adj.shape[0]
    adj_out = adj.copy()
    ## find the position which need to be add
    adj_orig_dense = adj.todense()
    flag_adj = np.triu(adj_orig_dense, k=1)
    idx_list = np.argwhere(flag_adj == 1)
    selected_idx_of_idx_list = np.random.choice(len(idx_list), size=k, replace = False)
    selected_idx = idx_list[selected_idx_of_idx_list]
    adj_out[selected_idx[:, 0], selected_idx[:, 1]] = 0
    adj_out[selected_idx[:, 1], selected_idx[:, 0]] = 0

    return adj_out

def randomly_flip_features(features, k,seed):
    np.random.seed(seed)
    num_node = features.shape[0]
    num_features = features.shape[1]
    features_lil = features.tolil()
    flip_node_idx_select = np.random.choice(num_node, size = 3000, replace = False)   ## select 100 node
    flip_node_idx = np.random.choice(flip_node_idx_select, size=k,replace = True)
    flip_fea_idx_select = np.random.choice(num_features, size = 10, replace = False)   ## select 2 features
    flip_fea_idx = np.random.choice(flip_fea_idx_select, size=k)
    ### this is the matrix one
    for i in range(len(flip_node_idx)):
        if features[flip_node_idx[i], flip_fea_idx[i]] == 1:
            features_lil[flip_node_idx[i], flip_fea_idx[i]] = 0
        else:
           features_lil[flip_node_idx[i], flip_fea_idx[i]] = 1
    return features_lil.tocsr()

if __name__ == "__main__":
    dense = np.diag(np.random.randint(1,100, size = 1000))
    features = sp.csr_matrix(dense)
    randomly_flip_features(features, 1000, 142)
