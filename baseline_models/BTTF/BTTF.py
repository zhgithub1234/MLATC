import numpy as np
from numpy.linalg import inv as inv
from numpy.random import normal as normrnd
from scipy.linalg import khatri_rao as kr_prod
from scipy.stats import wishart
from scipy.stats import invwishart
from numpy.linalg import solve as solve
from numpy.linalg import cholesky as cholesky_lower
from scipy.linalg import cholesky as cholesky_upper
from scipy.linalg import solve_triangular as solve_ut
import matplotlib.pyplot as plt
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

def mat2ten(mat, tensor_size, mode):
    index = list()
    index.append(mode)
    for i in range(tensor_size.shape[0]):
        if i != mode:
            index.append(i)
    return np.moveaxis(np.reshape(mat, tensor_size[index].tolist(), order='F'), 0, mode)

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


def mvnrnd_pre(mu, lambda_matrix):
    src = normrnd(size=(mu.shape[0],))
    return solve_ut(cholesky_upper(lambda_matrix, overwrite_a=True, check_finite=False),
                    src, lower=False, check_finite=False, overwrite_b=True) + mu


def cov_mat(mat, mat_bar):
    mat = mat - mat_bar
    return mat.T @ mat


def ten2mat(tensor, mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order='F')


def sample_factor_u(tau_sparse_tensor, tau_ind, u_matrix, v_matrix, x_tensor, beta0=1):
    """Sampling M-by-R factor matrix U and its hyperparameters (mu_u, Lambda_u)."""

    dim1, rank = u_matrix.shape
    u_bar = np.mean(u_matrix, axis=0)
    temp = dim1 / (dim1 + beta0)
    var_mu_hyper = temp * u_bar
    var_u_hyper = inv(np.eye(rank) + cov_mat(u_matrix, u_bar) + temp * beta0 * np.outer(u_bar, u_bar))
    var_lambda_hyper = wishart.rvs(df=dim1 + rank, scale=var_u_hyper)
    var_mu_hyper = mvnrnd_pre(var_mu_hyper, (dim1 + beta0) * var_lambda_hyper)

    var1 = kr_prod(x_tensor, v_matrix).T
    var2 = kr_prod(var1, var1)
    var3 = (var2 @ ten2mat(tau_ind, 0).T).reshape([rank, rank, dim1]) + var_lambda_hyper[:, :, None]
    var4 = var1 @ ten2mat(tau_sparse_tensor, 0).T + (var_lambda_hyper @ var_mu_hyper)[:, None]
    for i in range(dim1):
        u_matrix[i, :] = mvnrnd_pre(solve(var3[:, :, i], var4[:, i]), var3[:, :, i])

    return u_matrix


def sample_factor_v(tau_sparse_tensor, tau_ind, u_matrix, v_matrix, x_tensor, beta0=1):
    """Sampling N-by-R factor matrix V and its hyperparameters (mu_v, Lambda_v)."""

    dim2, rank = v_matrix.shape
    v_bar = np.mean(v_matrix, axis=0)
    temp = dim2 / (dim2 + beta0)
    var_mu_hyper = temp * v_bar
    var_v_hyper = inv(np.eye(rank) + cov_mat(v_matrix, v_bar) + temp * beta0 * np.outer(v_bar, v_bar))
    var_lambda_hyper = wishart.rvs(df=dim2 + rank, scale=var_v_hyper)
    var_mu_hyper = mvnrnd_pre(var_mu_hyper, (dim2 + beta0) * var_lambda_hyper)

    var1 = kr_prod(x_tensor, u_matrix).T
    var2 = kr_prod(var1, var1)
    var3 = (var2 @ ten2mat(tau_ind, 1).T).reshape([rank, rank, dim2]) + var_lambda_hyper[:, :, None]
    var4 = var1 @ ten2mat(tau_sparse_tensor, 1).T + (var_lambda_hyper @ var_mu_hyper)[:, None]
    for j in range(dim2):
        v_matrix[j, :] = mvnrnd_pre(solve(var3[:, :, j], var4[:, j]), var3[:, :, j])

    return v_matrix


def mnrnd(m_matrix, u_matrix, v_matrix):
    """
    Generate matrix normal distributed random matrix.
    M is a m-by-n matrix, U is a m-by-m matrix, and V is a n-by-n matrix.
    """
    dim1, dim2 = m_matrix.shape
    x0 = np.random.randn(dim1, dim2)
    p_matrix = cholesky_lower(u_matrix)
    q_matrix = cholesky_lower(v_matrix)

    return m_matrix + p_matrix @ x0 @ q_matrix.T


def sample_var_coefficient(x_tensor, time_lags):
    dim, rank = x_tensor.shape
    d = time_lags.shape[0]
    tmax = np.max(time_lags)

    z_mat = x_tensor[tmax:dim, :]
    q_mat = np.zeros((dim - tmax, rank * d))
    for k in range(d):
        q_mat[:, k * rank:(k + 1) * rank] = x_tensor[tmax - time_lags[k]:dim - time_lags[k], :]
    var_psi0 = np.eye(rank * d) + q_mat.T @ q_mat
    var_psi = inv(var_psi0)
    var_m = var_psi @ q_mat.T @ z_mat
    var_s = np.eye(rank) + z_mat.T @ z_mat - var_m.T @ var_psi0 @ var_m
    sigma_matrix = invwishart.rvs(df=rank + dim - tmax, scale=var_s)

    return mnrnd(var_m, var_psi, sigma_matrix), sigma_matrix


def sample_factor_x(tau_sparse_tensor, tau_ind, time_lags, u_matrix, v_matrix, x_tensor, a_matrix, lambda_x):
    """Sampling T-by-R factor matrix X."""

    dim3, rank = x_tensor.shape
    tmax = np.max(time_lags)
    tmin = np.min(time_lags)
    d = time_lags.shape[0]
    a0 = np.dstack([a_matrix] * d)
    for k in range(d):
        a0[k * rank:(k + 1) * rank, :, k] = 0
    mat0 = lambda_x @ a_matrix.T
    mat1 = np.einsum('kij, jt -> kit', a_matrix.reshape([d, rank, rank]), lambda_x)
    mat2 = np.einsum('kit, kjt -> ij', mat1, a_matrix.reshape([d, rank, rank]))

    var1 = kr_prod(v_matrix, u_matrix).T
    var2 = kr_prod(var1, var1)
    var3 = (var2 @ ten2mat(tau_ind, 2).T).reshape([rank, rank, dim3]) + lambda_x[:, :, None]
    var4 = var1 @ ten2mat(tau_sparse_tensor, 2).T
    for t in range(dim3):
        mt = np.zeros((rank, rank))
        nt = np.zeros(rank)
        qt = mat0 @ x_tensor[t - time_lags, :].reshape(rank * d)
        index = list(range(0, d))
        if t >= dim3 - tmax and t < dim3 - tmin:
            index = list(np.where(t + time_lags < dim3)[0])
        elif t < tmax:
            qt = np.zeros(rank)
            index = list(np.where(t + time_lags >= tmax)[0])
        if t < dim3 - tmin:
            mt = mat2.copy()
            temp = np.zeros((rank * d, len(index)))
            n = 0
            for k in index:
                temp[:, n] = x_tensor[t + time_lags[k] - time_lags, :].reshape(rank * d)
                n += 1
            temp0 = x_tensor[t + time_lags[index], :].T - np.einsum('ijk, ik -> jk', a0[:, :, index], temp)
            nt = np.einsum('kij, jk -> i', mat1[index, :, :], temp0)

        var3[:, :, t] = var3[:, :, t] + mt
        if t < tmax:
            var3[:, :, t] = var3[:, :, t] - lambda_x + np.eye(rank)
        x_tensor[t, :] = mvnrnd_pre(solve(var3[:, :, t], var4[:, t] + nt + qt), var3[:, :, t])

    return x_tensor


def sample_precision_tau(sparse_tensor, tensor_hat, ind):
    var_alpha = 1e-6 + 0.5 * np.sum(ind, axis=2)
    var_beta = 1e-6 + 0.5 * np.sum(((sparse_tensor - tensor_hat) ** 2) * ind, axis=2)
    return np.random.gamma(var_alpha, 1 / var_beta)


def compute_mape(var, var_hat):
    return np.sum(np.abs(var - var_hat) / var) / var.shape[0]


def compute_rmse(var, var_hat):
    return np.sqrt(np.sum((var - var_hat) ** 2) / var.shape[0])


def bttf(dense_tensor, sparse_tensor, init, rank, time_lags, burn_iter, gibbs_iter, multi_steps=1, vargin=0):
    """Bayesian Temporal Tensor Factorization, BTTF."""

    dim1, dim2, dim3 = sparse_tensor.shape
    d = time_lags.shape[0]
    u_matrix = init["U"]
    v_matrix = init["V"]
    x_tensor = init["X"]
    if np.isnan(sparse_tensor).any() == False:
        ind = sparse_tensor != 0
        pos_obs = np.where(ind)
        pos_test = np.where((dense_tensor != 0) & (sparse_tensor == 0))
    elif np.isnan(sparse_tensor).any() == True:
        pos_test = np.where((dense_tensor != 0) & (np.isnan(sparse_tensor)))
        ind = ~np.isnan(sparse_tensor)
        pos_obs = np.where(ind)
        sparse_tensor[np.isnan(sparse_tensor)] = 0
    dense_test = dense_tensor[pos_test]
    u_plus = np.zeros((dim1, rank))
    v_plus = np.zeros((dim2, rank))
    x_new_plus = np.zeros((dim3 + multi_steps, rank))
    a_plus = np.zeros((rank * d, rank))
    temp_hat = np.zeros(sparse_tensor.shape)
    show_iter = 200
    if vargin == 0:  # scalar tau
        tau = 1
    elif vargin == 1:  # matrix tau
        tau = np.ones((dim1, dim2))
    tensor_hat_plus = np.zeros(sparse_tensor.shape)
    for it in range(burn_iter + gibbs_iter):
        if vargin == 0:  # scalar tau
            tau_ind = tau * ind
            tau_sparse_tensor = tau * sparse_tensor
            u_matrix = sample_factor_u(tau_sparse_tensor, tau_ind, u_matrix, v_matrix, x_tensor)
            v_matrix = sample_factor_v(tau_sparse_tensor, tau_ind, u_matrix, v_matrix, x_tensor)
            a_matrix, sigma_matrix = sample_var_coefficient(x_tensor, time_lags)
            x_tensor = sample_factor_x(tau_sparse_tensor, tau_ind, time_lags, u_matrix, v_matrix, x_tensor, a_matrix, inv(sigma_matrix))
            tensor_hat = np.einsum('is, js, ts -> ijt', u_matrix, v_matrix, x_tensor)
            tau = np.random.gamma(1e-6 + 0.5 * np.sum(ind),
                                  1 / (1e-6 + 0.5 * np.sum(((sparse_tensor - tensor_hat) ** 2) * ind)))
        elif vargin == 1:  # matrix tau
            tau_ind = tau[:, :, None] * ind
            tau_sparse_tensor = tau[:, :, None] * sparse_tensor
            u_matrix = sample_factor_u(tau_sparse_tensor, tau_ind, u_matrix, v_matrix, x_tensor)
            v_matrix = sample_factor_v(tau_sparse_tensor, tau_ind, u_matrix, v_matrix, x_tensor)
            a_matrix, sigma_matrix = sample_var_coefficient(x_tensor, time_lags)
            x_tensor = sample_factor_x(tau_sparse_tensor, tau_ind, time_lags, u_matrix, v_matrix, x_tensor, a_matrix, inv(sigma_matrix))
            tensor_hat = np.einsum('is, js, ts -> ijt', u_matrix, v_matrix, x_tensor)
            tau = sample_precision_tau(sparse_tensor, tensor_hat, ind)
        temp_hat += tensor_hat
        if (it + 1) % show_iter == 0 and it < burn_iter:
            temp_hat = temp_hat / show_iter
            print('Iter: {}'.format(it + 1))
            print('MAPE: {:.6}'.format(compute_mape(dense_test, temp_hat[pos_test])))
            print('RMSE: {:.6}'.format(compute_rmse(dense_test, temp_hat[pos_test])))
            temp_hat = np.zeros(sparse_tensor.shape)
            print()
        x_new = np.zeros((dim3 + multi_steps, rank))
        if it + 1 > burn_iter:
            u_plus += u_matrix
            v_plus += v_matrix
            a_plus += a_matrix
            x_new[:dim3, :] = x_tensor.copy()
            if multi_steps == 1:
                x_new[dim3, :] = a_matrix.T @ x_new[dim3 - time_lags, :].reshape(rank * d)
            elif multi_steps > 1:
                for t0 in range(multi_steps):
                    x_new[dim3 + t0, :] = a_matrix.T @ x_new[dim3 + t0 - time_lags, :].reshape(rank * d)
            x_new_plus += x_new
            tensor_hat_plus += tensor_hat
    tensor_hat = tensor_hat_plus / gibbs_iter
    u_matrix = u_plus / gibbs_iter
    v_matrix = v_plus / gibbs_iter
    x_new = x_new_plus / gibbs_iter
    a_matrix = a_plus / gibbs_iter
    res_mape = compute_mape(dense_tensor[pos_test], tensor_hat[pos_test])
    res_rmse = compute_rmse(dense_tensor[pos_test], tensor_hat[pos_test])

    return tensor_hat, u_matrix, v_matrix, x_new, a_matrix, res_mape, res_rmse
