import numpy as np
from numpy.random import multivariate_normal as mvnrnd
from scipy.stats import wishart
from numpy.random import normal as normrnd
from scipy.linalg import khatri_rao as kr_prod
from numpy.linalg import inv as inv
from numpy.linalg import solve as solve
from numpy.linalg import cholesky as cholesky_lower
from scipy.linalg import cholesky as cholesky_upper
from scipy.linalg import solve_triangular as solve_ut
import time
import scipy.io

import csv
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
results_root = os.path.join(project_root, 'results')

if project_root not in sys.path:
    sys.path.append(project_root)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def mae(xhat, xtrue, mask):
    mask = mask.astype(bool)
    xhat = np.array(xhat)
    xtrue = np.array(xtrue)
    return np.mean(np.abs(xhat - xtrue)[~mask])

def rmse(xhat, xtrue, mask):
    mask = mask.astype(bool)
    xhat = np.array(xhat)
    xtrue = np.array(xtrue)
    return np.sqrt(np.mean(np.power(xhat - xtrue, 2)[~mask]))

def mape(xhat, xtrue, mask):
    mask = mask.astype(bool)
    xhat = np.array(xhat)
    xtrue = np.array(xtrue)
    return np.mean(np.abs((xhat - xtrue) / xtrue)[~mask])

def mvnrnd_pre(mu, lambda_matrix):
    src = normrnd(size=(mu.shape[0],))
    return solve_ut(cholesky_upper(lambda_matrix, overwrite_a=True, check_finite=False),
                    src, lower=False, check_finite=False, overwrite_b=True) + mu


def cp_combine(var):
    return np.einsum('is, js, ts -> ijt', var[0], var[1], var[2])


def ten2mat(tensor, mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order='F')


def cov_mat(mat, mat_bar):
    mat = mat - mat_bar
    return mat.T @ mat

def mat2ten(mat, tensor_size, mode):
    index = list()
    index.append(mode)
    for i in range(tensor_size.shape[0]):
        if i != mode:
            index.append(i)
    return np.moveaxis(np.reshape(mat, tensor_size[index].tolist(), order='F'), 0, mode)

def sample_factor(tau_sparse_tensor, tau_ind, factor, k, beta0=1):
    dim, rank = factor[k].shape
    factor_bar = np.mean(factor[k], axis=0)
    temp = dim / (dim + beta0)
    var_mu_hyper = temp * factor_bar
    var_w_hyper = inv(np.eye(rank) + cov_mat(factor[k], factor_bar) + temp * beta0 * np.outer(factor_bar, factor_bar))
    var_lambda_hyper = wishart.rvs(df=dim + rank, scale=var_w_hyper)
    var_mu_hyper = mvnrnd_pre(var_mu_hyper, (dim + beta0) * var_lambda_hyper)

    idx = list(filter(lambda x: x != k, range(len(factor))))
    var1 = kr_prod(factor[idx[1]], factor[idx[0]]).T
    var2 = kr_prod(var1, var1)
    var3 = (var2 @ ten2mat(tau_ind, k).T).reshape([rank, rank, dim]) + var_lambda_hyper[:, :, np.newaxis]
    var4 = var1 @ ten2mat(tau_sparse_tensor, k).T + (var_lambda_hyper @ var_mu_hyper)[:, np.newaxis]
    for i in range(dim):
        factor[k][i, :] = mvnrnd_pre(solve(var3[:, :, i], var4[:, i]), var3[:, :, i])
    return factor[k]


def sample_precision_tau(sparse_tensor, tensor_hat, ind):
    var_alpha = 1e-6 + 0.5 * np.sum(ind)
    var_beta = 1e-6 + 0.5 * np.sum(((sparse_tensor - tensor_hat) ** 2) * ind)
    return np.random.gamma(var_alpha, 1 / var_beta)


def compute_mape(var, var_hat):
    return np.sum(np.abs(var - var_hat) / var) / var.shape[0]


def compute_rmse(var, var_hat):
    return np.sqrt(np.sum((var - var_hat) ** 2) / var.shape[0])


def save_result_to_csv(model_name, dataset_name, missing_type, missing_rate, rank, iter_time, reslut_mape, reslut_rmse):
    csv_file_path = os.path.join(results_root, 'model_results', dataset_name, model_name)
    if not os.path.exists(csv_file_path):
        os.makedirs(csv_file_path)

    csv_path = os.path.join(csv_file_path, 'output.csv')
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(['Missing_type', 'Missing_rate', 'Rank', 'Iter_time', 'MAPE', 'RMSE'])
        reslut_mape = round(reslut_mape * 100, 2)
        reslut_rmse = round(reslut_rmse, 2)
        iter_time = round(iter_time, 2)

        writer.writerow([missing_type, missing_rate, rank, iter_time, reslut_mape, reslut_rmse])


def bgcp(dense_tensor, sparse_tensor, factor, burn_iter, gibbs_iter):
    """Bayesian Gaussian CP (BGCP) decomposition."""

    dim = np.array(sparse_tensor.shape)
    rank = factor[0].shape[1]

    if np.isnan(sparse_tensor).any() == False:
        ind = sparse_tensor != 0
        pos_obs = np.where(ind)
        pos_test = np.where((dense_tensor != 0) & (sparse_tensor == 0))
    elif np.isnan(sparse_tensor).any() == True:
        pos_test = np.where((dense_tensor != 0) & (np.isnan(sparse_tensor)))
        ind = ~np.isnan(sparse_tensor)
        pos_obs = np.where(ind)
        sparse_tensor[np.isnan(sparse_tensor)] = 0

    show_iter = 200
    tau = 1
    factor_plus = []
    for k in range(len(dim)):
        factor_plus.append(np.zeros((dim[k], rank)))
    temp_hat = np.zeros(dim)
    tensor_hat_plus = np.zeros(dim)
    for it in range(burn_iter + gibbs_iter):
        tau_ind = tau * ind
        tau_sparse_tensor = tau * sparse_tensor

        for k in range(len(dim)):
            factor[k] = sample_factor(tau_sparse_tensor, tau_ind, factor, k)
        tensor_hat = cp_combine(factor)
        temp_hat += tensor_hat
        tau = sample_precision_tau(sparse_tensor, tensor_hat, ind)

        if it + 1 > burn_iter:
            factor_plus = [factor_plus[k] + factor[k] for k in range(len(dim))]
            tensor_hat_plus += tensor_hat
        if (it + 1) % show_iter == 0 and it < burn_iter:
            temp_hat = temp_hat / show_iter
            print('Iter: {}'.format(it + 1))
            print('MAPE: {:.6}'.format(compute_mape(dense_tensor[pos_test], temp_hat[pos_test])))
            print('RMSE: {:.6}'.format(compute_rmse(dense_tensor[pos_test], temp_hat[pos_test])))
            temp_hat = np.zeros(sparse_tensor.shape)
            print()
    factor = [i / gibbs_iter for i in factor_plus]
    tensor_hat = tensor_hat_plus / gibbs_iter
    res_mape = compute_mape(dense_tensor[pos_test], tensor_hat[pos_test])
    res_rmse = compute_rmse(dense_tensor[pos_test], tensor_hat[pos_test])

    return tensor_hat, factor, res_mape, res_rmse