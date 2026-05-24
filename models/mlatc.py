"""MLATC model implementation."""

from numpy.linalg import inv as inv
from numpy.random import normal as normrnd
from scipy import sparse

import numpy as np
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

def save_result_to_csv(model_name, dataset_name, model_type, missing_type, missing_rate, rho0, k_num, c, theta, epsilon, iter_num, iter_time, reslut_mape, reslut_rmse):
    csv_file_path = os.path.join(results_root, 'model_results', dataset_name, model_name)
    if not os.path.exists(csv_file_path):
        os.makedirs(csv_file_path)

    csv_path = os.path.join(csv_file_path, 'output.csv')
    prepare_result_csv(csv_path)
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write the header for a new result file.
        if file.tell() == 0:
            writer.writerow(result_columns)
        reslut_mape = round(reslut_mape * 100, 2)
        reslut_rmse = round(reslut_rmse, 2)
        iter_time = round(iter_time, 2)

        writer.writerow([
            missing_type,
            model_type,
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
        try:
            u, s, v = np.linalg.svd(mat @ mat.T, full_matrices=0)
        except np.linalg.LinAlgError as e:
            print(f"???????????: {e}")
            u, s, v = np.linalg.svd(mat @ mat.T + 1e-6 * np.eye(m), full_matrices=0)

        s = np.sqrt(s)
        idx = np.sum(s > tau)
        mid = np.zeros(idx)
        mid[:theta] = 1
        mid[theta: idx] = (s[theta: idx] - tau) / s[theta: idx]
        return (u[:, :idx] @ np.diag(mid)) @ (u[:, :idx].T @ mat)
    elif m > 2 * n:
        return svt_tnn(mat.T, tau, theta).T

    try:
        u, s, v = np.linalg.svd(mat, full_matrices=0)
    except np.linalg.LinAlgError as e:
        print(f"???????????: {e}")
        m, n = mat.shape
        u, s, v = np.linalg.svd(mat + 1e-6 * np.eye(m, n), full_matrices=0)

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

def mlatc(dataset_name, dense_tensor, sparse_tensor, alpha, rho0, c, theta, missing_type, missing_rate, k_num, start_time, model_type,
          epsilon_values, maxiter=100, num_inner_iterations=3, save_outputs=True, return_history=False,
          initial_a_matrix=None, return_initial_a_matrix=False):
    """Run MLATC imputation for one experiment setting."""

    epsilon = epsilon_values[-1] if epsilon_values else 1e-4
    lambda0 = c * rho0
    if model_type == 'MLATC-N-node':
        choosed_dim = 0
        a_adj = compute_node_correlation(dense_tensor, k_num)
    elif model_type == 'MLATC-T-time':
        choosed_dim = 1
        a_adj = compute_node_correlation(dense_tensor.transpose(1, 0, 2), k_num)
    elif model_type == 'MLATC-D-day':
        choosed_dim = 2
        a_adj = compute_node_correlation(dense_tensor.transpose(2, 1, 0), k_num)
    else:
        print('Invalid model type')

    top_k_neighbors = get_top_k_neighbors(a_adj, k_num)

    dim = np.array(sparse_tensor.shape)  # dimensions of tensor [N, I, J] node,timeofday,day
    dim_time = int(np.prod(dim) / dim[choosed_dim])  # np.prod(dim) = N*I*J dim_time = I*J
    sparse_mat = ten2mat(sparse_tensor, choosed_dim)
    pos_missing = np.where(sparse_mat == 0)
    pos_test = np.where((dense_tensor != 0) & (sparse_tensor == 0))
    dense_test = dense_tensor[pos_test]  # test data
    t_tensor = np.zeros(dim)
    z_tensor = sparse_tensor.copy()  # missing data tensor
    z_matrix = sparse_mat.copy()  # missing data matrix Z
    if initial_a_matrix is not None:
        a_matrix = initial_a_matrix.copy()
    elif missing_type == 'NodeM' or missing_type == 'TimeM' or missing_type == 'DayM' or missing_type == 'TM1' or missing_type == 'TM2' or missing_type == 'TM3':
        a_matrix = 0.001 * np.random.rand(1, k_num)
    else:
        a_matrix = 0.001 * np.random.rand(dim[choosed_dim], k_num)
    initial_a_matrix_used = a_matrix.copy()

    it = 0
    last_mat = sparse_mat.copy()
    snorm = np.linalg.norm(sparse_mat, 'fro')
    rho = rho0
    flag = True

    iter_tols = []
    iter_rmses = []

    iter_primal_residuals = []

    while True:

        for k in range(num_inner_iterations):

            rho = min(rho * 1.05, 1e5)
            tensor_hat = np.zeros(dim)
            for p in range(len(dim)):
                tensor_hat += alpha[p] * mat2ten(svt_tnn(ten2mat(z_tensor - t_tensor / rho, p), alpha[p] / rho, theta), dim, p)

            temp0 = rho / lambda0 * ten2mat(tensor_hat + t_tensor / rho, choosed_dim)
            mat = np.zeros((dim[choosed_dim], dim_time))

            if missing_type == 'NodeM' or missing_type == 'TimeM' or missing_type == 'DayM' or missing_type == 'TM1' or missing_type == 'TM2' or missing_type == 'TM3':
                for m in range(dim[choosed_dim]):
                    if flag:
                        # Match the legacy MLATC initialization for structured missing cases.
                        mat[m, :] = np.mean(z_matrix[top_k_neighbors[m, :], :])
                    else:
                        z_neighbors = z_matrix[top_k_neighbors[m, :], :]
                        temp0_up = a_matrix[0, :] @ z_neighbors + temp0[m, :]
                        temp0_down = rho / lambda0 + 1
                        mat[m, :] = temp0_up / temp0_down
            else:
                for m in range(dim[choosed_dim]):
                    z_neighbors = z_matrix[top_k_neighbors[m, :], :]
                    temp0_up = a_matrix[m, :] @ z_neighbors + temp0[m, :]
                    temp0_down = rho / lambda0 + 1
                    mat[m, :] = temp0_up / temp0_down

            flag = False
            z_matrix[pos_missing] = mat[pos_missing]
            z_tensor = mat2ten(z_matrix, dim, choosed_dim)
            t_tensor = t_tensor + rho * (tensor_hat - z_tensor)


        if missing_type == 'NodeM' or missing_type == 'TimeM' or missing_type == 'DayM' or missing_type == 'TM1' or missing_type == 'TM2' or missing_type == 'TM3':
            temp_a = np.zeros((dim[choosed_dim], k_num))
            for m in range(dim[choosed_dim]):
                z_neighbors = z_matrix[top_k_neighbors[m, :], :]
                temp_a[m, :] = np.linalg.lstsq(z_neighbors.T, z_matrix[m, :], rcond=None)[0]

            filtered_temp_a = np.where((temp_a > 10) | (temp_a < -10), np.nan, temp_a)
            a_matrix = np.empty((1, temp_a.shape[1]))
            a_matrix[0, :] = np.nanmean(filtered_temp_a, axis=0)
        else:
            for m in range(dim[choosed_dim]):
                z_neighbors = z_matrix[top_k_neighbors[m, :], :]
                a_matrix[m, :] = np.linalg.lstsq(z_neighbors.T, z_matrix[m, :], rcond=None)[0]


        mat_hat = ten2mat(tensor_hat, choosed_dim)
        tol = np.linalg.norm((mat_hat - last_mat), 'fro') / snorm
        last_mat = mat_hat
        it += 1

        it_rmse = compute_rmse(dense_test, tensor_hat[pos_test])
        it_rmse = round(it_rmse, 2)
        iter_tols.append(tol)
        iter_rmses.append(it_rmse)

        primal_residual = np.linalg.norm((mat_hat - z_matrix), 'fro')
        iter_primal_residuals.append(primal_residual)

        if it >= maxiter or tol < epsilon:
            break

    end_time = time.time()
    print_result(it, tol, dense_test, tensor_hat[pos_test])
    reslut_mape = compute_mape(dense_test, tensor_hat[pos_test])
    reslut_rmse = compute_rmse(dense_test, tensor_hat[pos_test])
    iter_time = end_time - start_time

    history = pd.DataFrame({
        'Iteration': range(1, it + 1),
        'Tol': iter_tols,
        'RMSE': iter_rmses,
        'Primal_residuals': iter_primal_residuals,
    })

    if save_outputs:
        model_name = 'MLATC'
        csv_file_path = os.path.join(results_root, 'model_results', dataset_name, model_name + '_iter')
        if not os.path.exists(csv_file_path):
            os.makedirs(csv_file_path)

        csv_filename = "{}_{}_{}_{}_{}_{}_{}_tol_rmse.csv".format(
            missing_type,
            format_saved_number(missing_rate),
            format_saved_number(epsilon),
            format_saved_number(rho0),
            format_saved_number(k_num),
            format_saved_number(c),
            format_saved_number(theta),
        )
        csv_path = os.path.join(csv_file_path, csv_filename)

        history.to_csv(csv_path, index=False)

        save_result_to_csv(model_name, dataset_name, model_type, missing_type, missing_rate, rho0, k_num, c, theta, epsilon, it, iter_time, reslut_mape, reslut_rmse)

    if return_history and return_initial_a_matrix:
        return tensor_hat, reslut_rmse, history, initial_a_matrix_used
    if return_history:
        return tensor_hat, reslut_rmse, history
    if return_initial_a_matrix:
        return tensor_hat, reslut_rmse, initial_a_matrix_used
    return tensor_hat, reslut_rmse

def run_mlatc_with_configurations(datasets_paths, configurations, dataset_name, model_type, c, theta, missing_type=None, missing_rate=None, epsilon_values=None, rho=None, k_num=None):
    data_path = datasets_paths[dataset_name]
    if 'Guangzhou' in dataset_name or 'Hangzhou' in dataset_name:
        dense_tensor = scipy.io.loadmat(data_path)['tensor'].transpose(0, 2, 1)
    elif 'Seattle' in dataset_name:
        dense_tensor = np.load(data_path)['arr_0'].transpose(0, 2, 1)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dim1, dim2, dim3 = dense_tensor.shape

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
        # guangzhou 6; Seattle-data-set/tensor.npz 12
        block_window = 6
        vec = np.random.rand(int(dim_time / block_window))
        temp = np.array([vec] * block_window)
        vec = temp.reshape([dim2 * dim3], order='F')
        sparse_tensor = mat2ten(ten2mat(dense_tensor, 0) * np.round(vec + 0.5 - missing_rate)[None, :],
                                np.array([dim1, dim2, dim3]), 0)

    else:
        raise ValueError(f"Unknown missing type: {missing_type}")

    start_time = time.time()
    tensor_hat, reslut_rmse = mlatc(dataset_name, dense_tensor, sparse_tensor, alpha, rho, c, theta,
                                    missing_type, missing_rate, k_num, start_time, model_type, epsilon_values)

    return sparse_tensor, reslut_rmse

def mlatc_choose_parameters(c, theta, model_type_num, dataset_name, missing_type, missing_rate, epsilon_values, rho, k_num):
    np.random.seed(1000)
    datasets_paths = {
        'Guangzhou': os.path.join(datasets_root, 'Guangzhou-data-set', 'tensor.mat'),
        'Hangzhou': os.path.join(datasets_root, 'Hangzhou-data-set', 'tensor.mat'),
        'Seattle': os.path.join(datasets_root, 'Seattle-data-set', 'tensor.npz')
    }
    configurations = {
        'Missing':  {
            'alpha': np.ones(3) / 3,
        },
    }
    model_types = ['MLATC-N-node', 'MLATC-T-time', 'MLATC-D-day']
    model_type = model_types[model_type_num]

    tensor_hat, reslut_rmse = run_mlatc_with_configurations(datasets_paths, configurations, dataset_name, model_type, c, theta, missing_type, missing_rate, epsilon_values, rho, k_num)

    return tensor_hat, reslut_rmse

