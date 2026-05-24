import numpy as np
import time
import scipy.io
import pandas as pd
import os
import csv

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
results_root = os.path.join(project_root, 'results')

def ten2mat(tensor, mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order='F')

def mat2ten(mat, tensor_size, mode):
    index = list()
    index.append(mode)
    for i in range(tensor_size.shape[0]):
        if i != mode:
            index.append(i)
    return np.moveaxis(np.reshape(mat, list(tensor_size[index]), order='F'), 0, mode)

def svt_tnn(mat, alpha, rho, theta):
    tau = alpha / rho
    [m, n] = mat.shape
    if 2 * m < n:
        u, s, v = np.linalg.svd(mat @ mat.T, full_matrices=0)
        s = np.sqrt(s)
        idx = np.sum(s > tau)
        mid = np.zeros(idx)
        mid[:theta] = 1
        mid[theta:idx] = (s[theta:idx] - tau) / s[theta:idx]
        return (u[:, :idx] @ np.diag(mid)) @ (u[:, :idx].T @ mat)
    elif m > 2 * n:
        return svt_tnn(mat.T, alpha, rho, theta).T
    u, s, v = np.linalg.svd(mat, full_matrices=0)
    idx = np.sum(s > tau)
    vec = s[:idx].copy()
    vec[theta:idx] = s[theta:idx] - tau
    return u[:, :idx] @ np.diag(vec) @ v[:idx, :]

def compute_rmse(var, var_hat):
    return np.sqrt(np.sum((var - var_hat) ** 2) / var.shape[0])

def compute_mape(var, var_hat):
    return np.sum(np.abs(var - var_hat) / var) / var.shape[0]

def save_result_to_csv(model_name, dataset_name, missing_type, missing_rate, theta, iter_time, reslut_mape, reslut_rmse):
    csv_file_path = os.path.join(results_root, 'model_results', dataset_name, model_name)
    if not os.path.exists(csv_file_path):
        os.makedirs(csv_file_path)

    csv_path = os.path.join(csv_file_path, 'output.csv')
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(['Missing_type', 'Missing_rate', 'Iter', 'Iter_time', 'MAPE', 'RMSE'])
        reslut_mape = round(reslut_mape * 100, 2)
        reslut_rmse = round(reslut_rmse, 2)
        iter_time = round(iter_time, 2)

        writer.writerow([missing_type, missing_rate, theta, iter_time, reslut_mape, reslut_rmse])

def lrtc(dense_tensor, sparse_tensor, alpha, rho, theta, epsilon, maxiter):
    """Low-Rank Tensor Completion with Truncated Nuclear Norm, LRTC-TNN."""
    dim = np.array(sparse_tensor.shape)
    pos_missing = np.where(sparse_tensor == 0)
    pos_test = np.where((dense_tensor != 0) & (sparse_tensor == 0))

    x_tensor = np.zeros(np.insert(dim, 0, len(dim)))  # \boldsymbol{\mathcal{X}}
    t_tensor = np.zeros(np.insert(dim, 0, len(dim)))  # \boldsymbol{\mathcal{T}}
    z_matrix = sparse_tensor.copy()
    last_tensor = sparse_tensor.copy()
    snorm = np.sqrt(np.sum(sparse_tensor ** 2))
    it = 0
    while True:
        rho = min(rho * 1.05, 1e5)
        for k in range(len(dim)):
            x_tensor[k] = mat2ten(svt_tnn(ten2mat(z_matrix - t_tensor[k] / rho, k), alpha[k], rho, int(np.ceil(theta * dim[k]))), dim, k)
        z_matrix[pos_missing] = np.mean(x_tensor + t_tensor / rho, axis=0)[pos_missing]
        t_tensor = t_tensor + rho * (x_tensor - np.broadcast_to(z_matrix, np.insert(dim, 0, len(dim))))
        tensor_hat = np.einsum('k, kmnt -> mnt', alpha, x_tensor)
        tol = np.sqrt(np.sum((tensor_hat - last_tensor) ** 2)) / snorm
        last_tensor = tensor_hat.copy()
        it += 1
        if (it + 1) % 50 == 0:
            print('Iter: {}'.format(it + 1))
            print('RMSE: {:.6}'.format(compute_rmse(dense_tensor[pos_test], tensor_hat[pos_test])))
            print()
        if (tol < epsilon) or (it >= maxiter):
            break

    res_mape = compute_mape(dense_tensor[pos_test], tensor_hat[pos_test])
    res_rmse = compute_rmse(dense_tensor[pos_test], tensor_hat[pos_test])

    return tensor_hat, res_mape, res_rmse

def main():
    np.random.seed(1000)

    for r in [0.1, 0.2, 0.3]:
        print('Missing rate = {}'.format(r))
        missing_rate = r

        dense_path = os.path.join(project_root, 'datasets', 'Guangzhou-data-set', 'tensor.mat')
        dense_tensor = scipy.io.loadmat(dense_path)['tensor'].transpose(0, 2, 1)
        dim1, dim2, dim3 = dense_tensor.shape
        sparse_tensor = dense_tensor * np.round(np.random.rand(dim3) + 0.5 - missing_rate)[None, None, :]

        start = time.time()
        alpha = np.ones(3) / 3
        rho = 1e-5
        theta = 0.25
        if r > 0.8:
            theta = 0.2
        epsilon = 1e-4
        maxiter = 100
        lrtc(dense_tensor, sparse_tensor, alpha, rho, theta, epsilon, maxiter)
        end = time.time()
        print('Running time: %d seconds' % (end - start))
        print()


    for r in [0.3, 0.7, 0.9]:
        print('Missing rate = {}'.format(r))
        missing_rate = r

        dense_path = os.path.join(project_root, 'datasets', 'Guangzhou-data-set', 'tensor.mat')
        dense_tensor = scipy.io.loadmat(dense_path)['tensor'].transpose(0, 2, 1)
        dim1, dim2, dim3 = dense_tensor.shape
        sparse_tensor = dense_tensor * np.round(np.random.rand(dim1, dim2, dim3) + 0.5 - missing_rate)

        start = time.time()
        alpha = np.ones(3) / 3
        rho = 1e-5
        theta = 0.25
        if r > 0.8:
            theta = 0.2
        epsilon = 1e-4
        maxiter = 100
        lrtc(dense_tensor, sparse_tensor, alpha, rho, theta, epsilon, maxiter)
        end = time.time()
        print('Running time: %d seconds' % (end - start))
        print()
def main_latc():

    np.random.seed(1000)
    model_name = 'LRTC-TNN_LATC'
    dataset_names = ['Guangzhou-data-set', 'Hangzhou-data-set']
    dataset_name = dataset_names[1]
    dense_path = os.path.join(project_root, 'datasets', dataset_name, 'tensor.mat')
    dense_tensor = scipy.io.loadmat(dense_path)['tensor'].transpose(0, 2, 1)
    dim1, dim2, dim3 = dense_tensor.shape

    alpha = np.ones(3) / 3
    rho = 1e-5
    epsilon = 1e-4
    maxiter = 100

    for missing_type in ['RM', 'BM', 'TM1', 'TM2', 'TM3']:
        for missing_rate in [0.3, 0.7, 0.9, 0.85, 0.95]:
            for theta in [0.05, 0.10, 0.15, 0.20, 0.25, 0.3]:
                if missing_type == 'RM':
                    sparse_tensor = dense_tensor * np.round(np.random.rand(dim1, dim2, dim3) + 0.5 - missing_rate)
                # Missing at all nodes for selected day/time pairs.
                elif missing_type == 'TM1':
                    sparse_tensor = dense_tensor * np.round(np.random.rand(dim2, dim3) + 0.5 - missing_rate)[None, :, :]
                # Missing at all times for selected node/day pairs.
                elif missing_type == 'TM2':
                    sparse_tensor = dense_tensor * np.round(np.random.rand(dim1, dim3) + 0.5 - missing_rate)[:, None, :]
                # Missing across all days for selected node/time pairs.
                elif missing_type == 'TM3':
                    sparse_tensor = dense_tensor * np.round(np.random.rand(dim1, dim2) + 0.5 - missing_rate)[:, :, None]
                elif missing_type == 'BM':
                    dim_time = dim2 * dim3
                    block_window = 6
                    vec = np.random.rand(int(dim_time / block_window))
                    temp = np.array([vec] * block_window)
                    vec = temp.reshape([dim2 * dim3], order='F')
                    sparse_tensor = mat2ten(ten2mat(dense_tensor, 0) * np.round(vec + 0.5 - missing_rate)[None, :],
                                            np.array([dim1, dim2, dim3]), 0)

                else:
                    raise ValueError(f"Unknown missing type: {missing_type}")

                start = time.time()
                tensor_hat, res_mape, res_rmse = lrtc(dense_tensor, sparse_tensor, alpha, rho, theta, epsilon, maxiter)
                end = time.time()
                print('Running time: %d seconds' % (end - start))
                iter_time = end - start
                save_result_to_csv(model_name, dataset_name, missing_type, missing_rate, theta, iter_time, res_mape, res_rmse)


    for missing_type in ['NodeM', 'TimeM', 'DayM']:
        for missing_rate in [0.1, 0.15, 0.2, 0.25, 0.3, 0.7, 0.9]:
            for theta in [0.05, 0.10, 0.15, 0.20, 0.25, 0.3]:
                if missing_type == 'NodeM':
                    sparse_tensor = dense_tensor * np.round(np.random.rand(dim1) + 0.5 - missing_rate)[:, None, None]
                elif missing_type == 'TimeM':
                    sparse_tensor = dense_tensor * np.round(np.random.rand(dim2) + 0.5 - missing_rate)[None, :, None]
                elif missing_type == 'DayM':
                    sparse_tensor = dense_tensor * np.round(np.random.rand(dim3) + 0.5 - missing_rate)[None, None, :]
                else:
                    raise ValueError(f"Unknown missing type: {missing_type}")

                start = time.time()
                tensor_hat, res_mape, res_rmse = lrtc(dense_tensor, sparse_tensor, alpha, rho, theta, epsilon, maxiter)
                end = time.time()
                print('Running time: %d seconds' % (end - start))
                iter_time = end - start
                save_result_to_csv(model_name, dataset_name, missing_type, missing_rate, theta, iter_time, res_mape, res_rmse)


if __name__ == '__main__':
    main_latc()
