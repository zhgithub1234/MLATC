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
from scipy import sparse
from scipy.sparse.linalg import spsolve as spsolve
import time
import scipy.io
import pandas as pd

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

def ten2mat(tensor, mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order='F')


def mat2ten(mat, tensor_size, mode):
    index = list()
    index.append(mode)
    for i in range(tensor_size.shape[0]):
        if i != mode:
            index.append(i)
    return np.moveaxis(np.reshape(mat, tensor_size[index].tolist(), order='F'), 0, mode)


def svt_tnn(mat, tau, theta):
    [m, n] = mat.shape
    if 2 * m < n:
        u, s, v = np.linalg.svd(mat @ mat.T, full_matrices=0)
        s = np.sqrt(s)
        idx = np.sum(s > tau)
        mid = np.zeros(idx)
        mid[:theta] = 1
        mid[theta: idx] = (s[theta: idx] - tau) / s[theta: idx]
        return (u[:, :idx] @ np.diag(mid)) @ (u[:, :idx].T @ mat)
    elif m > 2 * n:
        return svt_tnn(mat.T, tau, theta).T
    u, s, v = np.linalg.svd(mat, full_matrices=0)
    idx = np.sum(s > tau)
    vec = s[:idx].copy()
    vec[theta: idx] = s[theta: idx] - tau
    return u[:, :idx] @ np.diag(vec) @ v[:idx, :]


def compute_mape(var, var_hat):
    return np.sum(np.abs(var - var_hat) / var) / var.shape[0]


def compute_rmse(var, var_hat):
    return np.sqrt(np.sum((var - var_hat) ** 2) / var.shape[0])


def print_result(it, tol, var, var_hat):
    print('Iter: {}'.format(it))
    print('Tolerance: {:.6}'.format(tol))
    print('Imputation MAPE: {:.6}'.format(compute_mape(var, var_hat)))
    print('Imputation RMSE: {:.6}'.format(compute_rmse(var, var_hat)))
    print()

def save_result_to_csv(model_name, dataset_name, missing_type, missing_rate, rho0, lambda0, theta, epsilon, iter_num, iter_time, reslut_mape, reslut_rmse):
    csv_file_path = os.path.join(results_root, 'model_results', dataset_name, model_name)
    if not os.path.exists(csv_file_path):
        os.makedirs(csv_file_path)

    csv_path = os.path.join(csv_file_path, 'output.csv')
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(['Missing_type', 'Missing_rate', 'Rho0', 'lambda', 'theta', 'Epsilon', 'Iter_num', 'Iter_time', 'MAPE', 'RMSE'])
        reslut_mape = round(reslut_mape * 100, 2)
        reslut_rmse = round(reslut_rmse, 2)
        iter_time = round(iter_time, 2)

        writer.writerow([missing_type, missing_rate, rho0, lambda0, theta, epsilon, iter_num, iter_time, reslut_mape, reslut_rmse])

def generate_psi(dim_time, time_lags):
    """Build sparse lag matrices for autoregression."""

    psis = []
    max_lag = np.max(time_lags)
    for i in range(len(time_lags) + 1):
        row = np.arange(0, dim_time - max_lag)
        if i == 0:
            col = np.arange(0, dim_time - max_lag) + max_lag
        else:
            col = np.arange(0, dim_time - max_lag) + max_lag - time_lags[i - 1]
        data = np.ones(dim_time - max_lag)
        psi = sparse.coo_matrix((data, (row, col)), shape=(dim_time - max_lag, dim_time))
        psis.append(psi)
    return psis


def latc(dense_tensor, sparse_tensor, time_lags, alpha, rho0, lambda0, theta, missing_rate, missing_type, model_name, dataset_name, start,
         epsilon=1e-4, maxiter=100, num_inner_iterations=3):
    """Run LATC imputation for one experiment setting."""

    dim = np.array(sparse_tensor.shape)  # dimensions of tensor [N, I, J] node,timeofday,day
    dim_time = int(np.prod(dim) / dim[0])  # np.prod(dim) = N*I*J dim_time = I*J
    d = len(time_lags)
    max_lag = np.max(time_lags)
    sparse_mat = ten2mat(sparse_tensor, 0)
    pos_missing = np.where(sparse_mat == 0)
    pos_test = np.where((dense_tensor != 0) & (sparse_tensor == 0))
    dense_test = dense_tensor[pos_test] # test data
    del dense_tensor

    t_tensor = np.zeros(dim)
    z_tensor = sparse_tensor.copy() # missing data tensor
    z_matrix = sparse_mat.copy()  # missing data matrix Z
    a_matrix = 0.001 * np.random.rand(dim[0], d)
    psis = generate_psi(dim_time, time_lags)
    iden = sparse.coo_matrix((np.ones(dim_time), (np.arange(0, dim_time), np.arange(0, dim_time))),
                             shape=(dim_time, dim_time))
    it = 0
    ind = np.zeros((d, dim_time - max_lag), dtype=np.int_)
    for i in range(d):
        ind[i, :] = np.arange(max_lag - time_lags[i], dim_time - time_lags[i])
    last_mat = sparse_mat.copy()
    snorm = np.linalg.norm(sparse_mat, 'fro')
    rho = rho0
    while True:
        temp = []
        for m in range(dim[0]):
            psis0 = psis.copy()
            for i in range(d):
                psis0[i + 1] = a_matrix[m, i] * psis[i + 1]
            b_matrix = psis0[0] - sum(psis0[1:])
            temp.append(b_matrix.T @ b_matrix)
        
        for k in range(num_inner_iterations):

            rho = min(rho * 1.05, 1e5)
            tensor_hat = np.zeros(dim)
            for p in range(len(dim)):
                tensor_hat += alpha[p] * mat2ten(svt_tnn(ten2mat(z_tensor - t_tensor / rho, p), alpha[p] / rho, theta), dim, p)
            temp0 = rho / lambda0 * ten2mat(tensor_hat + t_tensor / rho, 0)
            mat = np.zeros((dim[0], dim_time))
            for m in range(dim[0]):
                mat[m, :] = spsolve(temp[m] + rho * iden / lambda0, temp0[m, :])
            z_matrix[pos_missing] = mat[pos_missing]
            z_tensor = mat2ten(z_matrix, dim, 0)
            t_tensor = t_tensor + rho * (tensor_hat - z_tensor)

        for m in range(dim[0]):
            a_matrix[m, :] = np.linalg.lstsq(z_matrix[m, ind].T, z_matrix[m, max_lag:], rcond=None)[0]
        mat_hat = ten2mat(tensor_hat, 0)
        tol = np.linalg.norm((mat_hat - last_mat), 'fro') / snorm
        last_mat = mat_hat.copy()
        it += 1
        if it % 10 == 0:
            print_result(it, tol, dense_test, tensor_hat[pos_test])
        if (tol < epsilon) or (it >= maxiter):
            break

    reslut_mape = compute_mape(dense_test, tensor_hat[pos_test])
    reslut_rmse = compute_rmse(dense_test, tensor_hat[pos_test])
    end = time.time()
    iter_time = end - start
    save_result_to_csv(model_name, dataset_name, missing_type, missing_rate, rho0, lambda0, theta, epsilon, it, iter_time, reslut_mape, reslut_rmse)

    return tensor_hat