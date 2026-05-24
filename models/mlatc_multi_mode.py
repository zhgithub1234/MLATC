"""MLATC multi-mode model implementation."""

import numpy as np
from numpy.linalg import inv as inv
from numpy.random import normal as normrnd
from scipy import sparse

import time
import scipy.io
import pandas as pd
import csv
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_root = os.path.join(project_root, 'results')
datasets_root = os.path.join(project_root, 'datasets')

if project_root not in sys.path:
    sys.path.append(project_root)

def format_saved_number(value):
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        text = f"{float(value):.6f}".rstrip('0').rstrip('.')
        return text if text else '0'
    return str(value)

result_columns = ['Missing_type', 'Model_type', 'Missing_rate', 'Rho0', 'K_num', 'c', 'theta', 'Epsilon', 'Iter_num', 'Iter_time', 'MAPE', 'RMSE']

def prepare_result_csv(csv_path):
    if not os.path.exists(csv_path):
        return

    with open(csv_path, mode='r', newline='') as file:
        current_header = next(csv.reader(file), [])

    if not current_header or current_header == result_columns:
        return

    base_path, extension = os.path.splitext(csv_path)
    backup_path = f"{base_path}_legacy_schema{extension}"
    backup_index = 1
    while os.path.exists(backup_path):
        backup_path = f"{base_path}_legacy_schema_{backup_index}{extension}"
        backup_index += 1

    # Backup obsolete result files before appending new rows.
    os.replace(csv_path, backup_path)

def save_result_to_csv(dataset_name, missing_type, missing_rate, rho0, k_num, c, theta, epsilon, iter_num, iter_time, reslut_mape, reslut_rmse):
    model_name = 'MLATC-Multi-mode'
    csv_file_path = os.path.join(results_root, 'model_results', dataset_name, model_name)
    if not os.path.exists(csv_file_path):
        os.makedirs(csv_file_path)

    csv_path = os.path.join(csv_file_path, 'output.csv')
    prepare_result_csv(csv_path)
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(result_columns)
        reslut_mape = round(reslut_mape * 100, 2)
        reslut_rmse = round(reslut_rmse, 2)
        iter_time = round(iter_time, 2)
        writer.writerow([
            missing_type,
            model_name,
            format_saved_number(missing_rate),
            format_saved_number(rho0),
            format_saved_number(k_num),
            format_saved_number(c),
            format_saved_number(theta),
            format_saved_number(epsilon),
            iter_num,
            iter_time,
            reslut_mape,
            reslut_rmse,
        ])

def compute_node_correlation(dense_tensor, k=6):
    """Build a top-k Pearson correlation mask."""
    reshaped_tensor = dense_tensor.reshape(dense_tensor.shape[0], -1)  # [nodenum, -1]

    correlation_matrix = np.corrcoef(reshaped_tensor)

    adj_matrix = np.zeros_like(correlation_matrix)

    for i in range(correlation_matrix.shape[0]):
        corr_row = correlation_matrix[i, :]
        corr_row[i] = -np.inf
        top_k_indices = np.argpartition(corr_row, -k)[-k:]
        adj_matrix[i, top_k_indices] = 1

    return adj_matrix

def get_top_k_neighbors(a_corr, k=6):
    dim = a_corr.shape[0]
    top_k_neighbors = np.zeros((dim, k), dtype=int)
    for i in range(dim):
        sorted_indices = np.argsort(a_corr[i])[-k:][::-1]
        sorted_nodes = sorted(sorted_indices)
        top_k_neighbors[i] = sorted_nodes
    return top_k_neighbors

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

def mlatc_multi_mode(dataset_name, dense_tensor, sparse_tensor, alpha, rho0, c, theta, missing_type, missing_rate, k_num, start_time, w0, w1, w2,
                     epsilon=1e-4, maxiter=100, num_inner_iterations=3):
    """Run MLATC multi-mode imputation for one experiment setting."""

    lambda0 = c * rho0

    dim = np.array(sparse_tensor.shape)  # dimensions of tensor [N, I, J] node,timeofday,day
    dim_0 = int(np.prod(dim) / dim[0])
    dim_1 = int(np.prod(dim) / dim[1])
    dim_2 = int(np.prod(dim) / dim[2])

    m_tensor = sparse_tensor.copy()
    t_tensor = np.zeros(dim)

    sparse_mat_0 = ten2mat(sparse_tensor, 0)
    sparse_mat_1 = ten2mat(sparse_tensor, 1)
    sparse_mat_2 = ten2mat(sparse_tensor, 2)
    pos_missing_0 = np.where(sparse_mat_0 == 0)
    pos_missing_1 = np.where(sparse_mat_1 == 0)
    pos_missing_2 = np.where(sparse_mat_2 == 0)
    pos_test = np.where((dense_tensor != 0) & (sparse_tensor == 0))
    dense_test = dense_tensor[pos_test]  # test data

    z_0 = sparse_mat_0.copy()
    z_1 = sparse_mat_1.copy()
    z_2 = sparse_mat_2.copy()
    t_0 = np.zeros(dim)
    t_1 = np.zeros(dim)
    t_2 = np.zeros(dim)
    a_adj_0 = compute_node_correlation(dense_tensor, k_num)
    a_adj_1 = compute_node_correlation(dense_tensor.transpose(1, 0, 2), k_num)
    a_adj_2 = compute_node_correlation(dense_tensor.transpose(2, 1, 0), k_num)

    del dense_tensor
    top_k_neighbors_0 = get_top_k_neighbors(a_adj_0, k_num)
    top_k_neighbors_1 = get_top_k_neighbors(a_adj_1, k_num)
    top_k_neighbors_2 = get_top_k_neighbors(a_adj_2, k_num)

    if missing_type == 'NodeM' or missing_type == 'TM1':
        a_0 = 0.001 * np.random.rand(1, k_num)
        a_1 = 0.001 * np.random.rand(1, k_num)
        a_2 = 0.001 * np.random.rand(1, k_num)
    else:
        a_0 = 0.001 * np.random.rand(dim[0], k_num)
        a_1 = 0.001 * np.random.rand(dim[1], k_num)
        a_2 = 0.001 * np.random.rand(dim[2], k_num)

    it = 0
    last_mat = sparse_mat_0.copy()
    snorm = np.linalg.norm(sparse_mat_0, 'fro')
    rho = rho0
    flag = True

    iter_tols = []
    iter_rmses = []

    while True:
        for k in range(num_inner_iterations):
            rho = min(rho * 1.05, 1e5)
            beta = 1e-3

            tensor_hat = np.zeros(dim)
            for p in range(len(dim)):
                tensor_hat += alpha[p] * mat2ten(svt_tnn(ten2mat(m_tensor - t_tensor / rho, p), alpha[p] / rho, theta), dim, p)

            m_up_left = rho * (tensor_hat + t_tensor / rho)

            w0 = w0 / (w0 + w1 + w2)
            w1 = w1 / (w0 + w1 + w2)
            w2 = w2 / (w0 + w1 + w2)

            m_up_right = w0 * beta * (mat2ten(z_0, dim, 0) - t_0 / beta)\
                         + w1 * beta * (mat2ten(z_1, dim, 1) - t_1 / beta)\
                         + w2 * beta * (mat2ten(z_2, dim, 2) - t_2 / beta)
            m_tensor = (m_up_left + m_up_right) / (rho + beta)

            temp0 = beta / lambda0 * ten2mat(m_tensor + t_0 / beta, 0)
            mat_0 = np.zeros((dim[0], dim_0))

            if missing_type == 'NodeM' or missing_type == 'TM1':
                for m in range(dim[0]):
                    if flag:
                        mat_0[m, :] = np.mean(z_0[top_k_neighbors_0[m, :], :])
                    else:
                        z_neighbors_0 = z_0[top_k_neighbors_0[m, :], :]
                        temp0_up = a_0[0, :] @ z_neighbors_0 + temp0[m, :]
                        temp0_down = beta / lambda0 + 1
                        mat_0[m, :] = temp0_up / temp0_down
            else:
                for m in range(dim[0]):
                    z_neighbors_0 = z_0[top_k_neighbors_0[m, :], :]
                    temp0_up = a_0[m, :] @ z_neighbors_0 + temp0[m, :]
                    temp0_down = beta / lambda0 + 1
                    mat_0[m, :] = temp0_up / temp0_down

            temp1 = beta / lambda0 * ten2mat(m_tensor + t_1 / beta, 1)
            mat_1 = np.zeros((dim[1], dim_1))

            if missing_type == 'NodeM' or missing_type == 'TM1':
                for m in range(dim[1]):
                    if flag:
                        mat_1[m, :] = np.mean(z_1[top_k_neighbors_1[m, :], :])
                    else:
                        z_neighbors_1 = z_1[top_k_neighbors_1[m, :], :]
                        temp1_up = a_1[0, :] @ z_neighbors_1 + temp1[m, :]
                        temp1_down = beta / lambda0 + 1
                        mat_1[m, :] = temp1_up / temp1_down
            else:
                for m in range(dim[1]):
                    z_neighbors_1 = z_1[top_k_neighbors_1[m, :], :]
                    temp1_up = a_1[m, :] @ z_neighbors_1 + temp1[m, :]
                    temp1_down = beta / lambda0 + 1
                    mat_1[m, :] = temp1_up / temp1_down

            temp2 = beta / lambda0 * ten2mat(m_tensor + t_2 / beta, 2)
            mat_2 = np.zeros((dim[2], dim_2))

            if missing_type == 'NodeM' or missing_type == 'TM1':
                for m in range(dim[2]):
                    if flag:
                        mat_2[m, :] = np.mean(z_2[top_k_neighbors_2[m, :], :])
                    else:
                        z_neighbors_2 = z_2[top_k_neighbors_2[m, :], :]
                        temp2_up = a_2[0, :] @ z_neighbors_2 + temp2[m, :]
                        temp2_down = beta / lambda0 + 1
                        mat_2[m, :] = temp2_up / temp2_down
            else:
                for m in range(dim[2]):
                    z_neighbors_2 = z_2[top_k_neighbors_2[m, :], :]
                    temp2_up = a_2[m, :] @ z_neighbors_2 + temp2[m, :]
                    temp2_down = beta / lambda0 + 1
                    mat_2[m, :] = temp2_up / temp2_down

            flag = False
            z_0[pos_missing_0] = mat_0[pos_missing_0]
            z_1[pos_missing_1] = mat_1[pos_missing_1]
            z_2[pos_missing_2] = mat_2[pos_missing_2]

            t_tensor = t_tensor + rho * (tensor_hat - m_tensor)
            t_0 = t_0 + beta * (m_tensor - mat2ten(mat_0, dim, 0))
            t_1 = t_1 + beta * (m_tensor - mat2ten(mat_1, dim, 1))
            t_2 = t_2 + beta * (m_tensor - mat2ten(mat_2, dim, 2))

        if missing_type == 'NodeM' or missing_type == 'TM1':
            temp_a_0 = np.zeros((dim[0], k_num))
            for m in range(dim[0]):
                z_neighbors_0 = z_0[top_k_neighbors_0[m, :], :]
                temp_a_0[m, :] = np.linalg.lstsq(z_neighbors_0.T, z_0[m, :], rcond=None)[0]

            filtered_temp_a = np.where((temp_a_0 > 10) | (temp_a_0 < -10), np.nan, temp_a_0)
            a_0 = np.empty((1, temp_a_0.shape[1]))
            a_0[0, :] = np.nanmean(filtered_temp_a, axis=0)
        else:
            for m in range(dim[0]):
                z_neighbors_0 = z_0[top_k_neighbors_0[m, :], :]
                a_0[m, :] = np.linalg.lstsq(z_neighbors_0.T, z_0[m, :], rcond=None)[0]

        if missing_type == 'NodeM' or missing_type == 'TM1':
            temp_a_1 = np.zeros((dim[1], k_num))
            for m in range(dim[1]):
                z_neighbors_1 = z_1[top_k_neighbors_1[m, :], :]
                temp_a_1[m, :] = np.linalg.lstsq(z_neighbors_1.T, z_1[m, :], rcond=None)[0]

            filtered_temp_a = np.where((temp_a_1 > 10) | (temp_a_1 < -10), np.nan, temp_a_1)
            a_1 = np.empty((1, temp_a_1.shape[1]))
            a_1[0, :] = np.nanmean(filtered_temp_a, axis=0)
        else:
            for m in range(dim[1]):
                z_neighbors_1 = z_1[top_k_neighbors_1[m, :], :]
                a_1[m, :] = np.linalg.lstsq(z_neighbors_1.T, z_1[m, :], rcond=None)[0]

        if missing_type == 'NodeM' or missing_type == 'TM1':
            temp_a_2 = np.zeros((dim[2], k_num))
            for m in range(dim[2]):
                z_neighbors_2 = z_2[top_k_neighbors_2[m, :], :]
                temp_a_2[m, :] = np.linalg.lstsq(z_neighbors_2.T, z_2[m, :], rcond=None)[0]

            filtered_temp_a = np.where((temp_a_2 > 10) | (temp_a_2 < -10), np.nan, temp_a_2)
            a_2 = np.empty((1, temp_a_2.shape[1]))
            a_2[0, :] = np.nanmean(filtered_temp_a, axis=0)
        else:
            for m in range(dim[2]):
                z_neighbors_2 = z_2[top_k_neighbors_2[m, :], :]
                a_2[m, :] = np.linalg.lstsq(z_neighbors_2.T, z_2[m, :], rcond=None)[0]


        mat_hat = ten2mat(tensor_hat, 0)
        tol = np.linalg.norm((mat_hat - last_mat), 'fro') / snorm
        last_mat = mat_hat.copy()
        it += 1

        it_rmse = compute_rmse(dense_test, tensor_hat[pos_test])
        it_rmse = round(it_rmse, 2)
        iter_tols.append(tol)
        iter_rmses.append(it_rmse)

        if it % 10 == 0:
            print_result(it, tol, dense_test, tensor_hat[pos_test])

        if (tol < epsilon) or (it >= maxiter):
            break

    end_time = time.time()
    print_result(it, tol, dense_test, tensor_hat[pos_test])
    reslut_mape = compute_mape(dense_test, tensor_hat[pos_test])
    reslut_rmse = compute_rmse(dense_test, tensor_hat[pos_test])
    iter_time = end_time - start_time

    model_name = 'MLATC-Multi-mode'
    csv_file_path = os.path.join(results_root, 'model_results', dataset_name, model_name + '_iter')
    if not os.path.exists(csv_file_path):
        os.makedirs(csv_file_path)

    csv_filename = "{}_{}_{}_{}_{}_{}_tol_rmse.csv".format(
        missing_type,
        format_saved_number(missing_rate),
        format_saved_number(epsilon),
        format_saved_number(k_num),
        format_saved_number(c),
        format_saved_number(theta),
    )
    csv_path = os.path.join(csv_file_path, csv_filename)

    df = pd.DataFrame({
        'Iteration': range(1, it + 1),
        'Tol': iter_tols,
        'RMSE': iter_rmses
    })
    df.to_csv(csv_path, index=False)

    save_result_to_csv(dataset_name, missing_type, missing_rate, rho0, k_num, c, theta, epsilon, it, iter_time, reslut_mape, reslut_rmse)

    return tensor_hat, reslut_rmse

def run_mlatc_multi_mode_with_configurations(datasets_paths, configurations, dataset_name, c, theta, missing_type, missing_rate, rho, k_num, w0, w1, w2):

    data_path = datasets_paths[dataset_name]
    if 'Guangzhou' in dataset_name or 'Hangzhou' in dataset_name:
        dense_tensor = scipy.io.loadmat(data_path)['tensor'].transpose(0, 2, 1)
    elif 'Seattle' in dataset_name:
        dense_tensor = np.load(data_path)['arr_0'].transpose(0, 2, 1)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dim1, dim2, dim3 = dense_tensor.shape

    # Use a single experiment setting per call.
    config = configurations['Missing']
    alpha = config['alpha']

    if missing_type == 'NodeM':
        sparse_tensor = dense_tensor * np.round(np.random.rand(dim1) + 0.5 - missing_rate)[:, None, None]
    elif missing_type == 'TimeM':
        sparse_tensor = dense_tensor * np.round(np.random.rand(dim2) + 0.5 - missing_rate)[None, :, None]
    elif missing_type == 'DayM':
        sparse_tensor = dense_tensor * np.round(np.random.rand(dim3) + 0.5 - missing_rate)[None, None, :]
    elif missing_type == 'RM':
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

    print(f"c: {c}, theta: {theta}, missing type: {missing_type}, missing rate: {missing_rate}")
    start_time = time.time()
    tensor_hat, reslut_rmse = mlatc_multi_mode(dataset_name, dense_tensor, sparse_tensor, alpha, rho, c, theta,
                                               missing_type, missing_rate, k_num, start_time, w0, w1, w2)

    return tensor_hat, reslut_rmse


def mlatc_multi_mode_choose_parameters(c, theta, dataset_name, missing_type, missing_rate, rho, k_num, w0, w1, w2):
    np.random.seed(1000)
    datasets_paths = {
        'Guangzhou': os.path.join(datasets_root, 'Guangzhou-data-set', 'tensor.mat'),
        'Hangzhou': os.path.join(datasets_root, 'Hangzhou-data-set', 'tensor.mat'),
        'Seattle': os.path.join(datasets_root, 'Seattle-data-set', 'tensor.npz')
    }
    configurations = {
        'Missing': {
            'alpha': np.ones(3) / 3
        }
    }

    tensor_hat, reslut_rmse = run_mlatc_multi_mode_with_configurations(
        datasets_paths, configurations, dataset_name, c, theta, missing_type, missing_rate, rho, k_num, w0, w1, w2
    )

    return tensor_hat, reslut_rmse


def choose_parameters(c, theta, dataset_name, missing_type, missing_rate, rho, k_num, w0, w1, w2):
    return mlatc_multi_mode_choose_parameters(c, theta, dataset_name, missing_type, missing_rate, rho, k_num, w0, w1, w2)


if __name__ == '__main__':
    dataset_name = 'Guangzhou'
    missing_type = 'RM'
    missing_rate = 0.3
    rho = 1e-5
    k_num = 3
    c_values = [0.1, 0.2, 1, 5, 10]
    theta_values = [5, 10, 15, 20, 25, 30]
    w0, w1, w2 = 0.33, 0.33, 0.34
    for c in c_values:
        for theta in theta_values:
            mlatc_multi_mode_choose_parameters(c, theta, dataset_name, missing_type, missing_rate, rho, k_num, w0, w1, w2)
