import os
import sys
from pathlib import Path

import numpy as np
import scipy.io
import time

project_root = Path(__file__).resolve().parents[1]
datasets_root = project_root / 'datasets'
project_results_root = project_root / 'results'

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models.mlatc import compute_mape, compute_rmse, mlatc
from baseline_models.BGCP.BGCP import bgcp
from baseline_models.BTTF.BTTF import bttf
from baseline_models.LRTC_TNN.LRTC_TNN import lrtc as lrtc_tnn
from baseline_models.LATC.LATC import latc
from utils.utils_for_results_save import *


def format_saved_number(value):
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        text = '{:.12f}'.format(float(value)).rstrip('0').rstrip('.')
        return text if text else '0'
    return str(value)


def calculate_imputation_metrics(dense_tensor, sparse_tensor, tensor_hat):
    pos_test = np.where((dense_tensor != 0) & (sparse_tensor == 0))
    dense_test = dense_tensor[pos_test]
    result_mape = round(compute_mape(dense_test, tensor_hat[pos_test]) * 100, 2)
    result_rmse = round(compute_rmse(dense_test, tensor_hat[pos_test]), 2)
    return result_mape, result_rmse


def save_best_parameters_to_npy(best_parameters):
    parameter_dir = (
        project_results_root
        / 'model_results'
        / best_parameters['dataset_name']
        / best_parameters['model_name']
        / 'best_parameters'
    )
    parameter_dir.mkdir(parents=True, exist_ok=True)
    filename = '{}_{}.npy'.format(
        best_parameters['missing_type'],
        format_saved_number(best_parameters['missing_rate']),
    )
    parameter_path = parameter_dir / filename
    np.save(str(parameter_path), best_parameters, allow_pickle=True)
    print('Saved best parameters to {}'.format(parameter_path))


def save_current_best_parameters(best_result):
    best_parameters = best_result.copy()
    best_parameters.pop('best_rmse_for_selection', None)
    save_best_parameters_to_npy(best_parameters)



def ten2mat(tensor, mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order='F')

def mat2ten(mat, tensor_size, mode):
    index = list()
    index.append(mode)
    for i in range(tensor_size.shape[0]):
        if i != mode:
            index.append(i)
    return np.moveaxis(np.reshape(mat, tensor_size[index].tolist(), order='F'), 0, mode)


np.random.seed(1000)
datasets_paths = {
    'Guangzhou': str(datasets_root / 'Guangzhou-data-set' / 'tensor.mat'),
    'Hangzhou': str(datasets_root / 'Hangzhou-data-set' / 'tensor.mat'),
    'Seattle': str(datasets_root / 'Seattle-data-set' / 'tensor.npz'),
    }

dataset_names = ['Hangzhou']
model_names = ['MLATC']
missing_types = ['RM']
missing_rates_set = {
        "RM": [0.3, 0.7, 0.9],
        "BM": [0.3, 0.7, 0.9],
        "TM1": [0.3, 0.7, 0.9],
        "TM2": [0.3, 0.7, 0.9],
        "TM3": [0.3, 0.7, 0.9],
        "NodeM": [0.3, 0.7, 0.9],
        "TimeM": [0.3, 0.7, 0.9],
        "DayM": [0.3, 0.7, 0.9],
    }

for dataset_name in dataset_names:
    for model_name in model_names:
        for missing_type in missing_types:
            missing_rates = missing_rates_set[missing_type]
            for missing_rate in missing_rates:
                data_path = datasets_paths[dataset_name]
                if 'Guangzhou' in dataset_name or 'Hangzhou' in dataset_name:
                    dense_tensor = scipy.io.loadmat(data_path)['tensor'].transpose(0, 2, 1)
                elif 'Seattle' in dataset_name:
                    dense_tensor = np.load(data_path)['arr_0'].transpose(0, 2, 1)

                else:
                    raise ValueError(f"Unknown dataset: {dataset_name}")

                dim1, dim2, dim3 = dense_tensor.shape

                missing_data_dir = datasets_root / 'generated_missing_data' / dataset_name / missing_type
                missing_data_path = missing_data_dir / f'{missing_rate}.npy'
                if not missing_data_dir.exists():
                    missing_data_dir.mkdir(parents=True)

                if missing_data_path.exists():
                    sparse_tensor = np.load(str(missing_data_path))
                    print(f"Loading existing missing data for {dataset_name}, {missing_type}, {missing_rate}")
                else:

                    if missing_type == 'NodeM':
                        sparse_tensor = dense_tensor * np.round(np.random.rand(dim1) + 0.5 - missing_rate)[:, None, None]
                    elif missing_type == 'TimeM':
                        sparse_tensor = dense_tensor * np.round(np.random.rand(dim2) + 0.5 - missing_rate)[None, :, None]
                    elif missing_type == 'DayM':
                        sparse_tensor = dense_tensor * np.round(np.random.rand(dim3) + 0.5 - missing_rate)[None, None, :]
                    elif missing_type == 'RM':
                        sparse_tensor = dense_tensor * np.round(np.random.rand(dim1, dim2, dim3) + 0.5 - missing_rate)
                    elif missing_type == 'NM':
                        sparse_tensor = dense_tensor * np.round(np.random.rand(dim1, dim3) + 0.5 - missing_rate)[:, None, :]
                    elif missing_type == 'TM1':
                        sparse_tensor = dense_tensor * np.round(np.random.rand(dim2, dim3) + 0.5 - missing_rate)[None, :, :]
                    elif missing_type == 'TM2':
                        sparse_tensor = dense_tensor * np.round(np.random.rand(dim1, dim3) + 0.5 - missing_rate)[:, None, :]
                    elif missing_type == 'TM3':
                        sparse_tensor = dense_tensor * np.round(np.random.rand(dim1, dim2) + 0.5 - missing_rate)[:, :, None]
                    elif missing_type == 'BM':
                        dim_time = dim2 * dim3
                        block_window = 6
                        vec = np.random.rand(int(dim_time / block_window))
                        temp = np.array([vec] * block_window)
                        vec = temp.reshape([dim2 * dim3], order='F')
                        sparse_tensor = mat2ten(ten2mat(dense_tensor, 0) * np.round(vec + 0.5 - missing_rate)[None, :], np.array([dim1, dim2, dim3]), 0)

                    else:
                        raise ValueError(f"Unknown missing type: {missing_type}")

                    np.save(str(missing_data_path), sparse_tensor)

                if model_name == 'MLATC':

                    model_types = ['MLATC-N-node', 'MLATC-T-time', 'MLATC-D-day']
                    epsilon_values = [1e-4]

                    rho_values = [1e-5, 5e-5, 1e-4]
                    c_values = [0.1, 0.2, 1, 5, 10]
                    theta_values = [5, 10, 15, 20, 25, 30]
                    k_num_values = [3, 6, 9, 12]
                    m_values = [0]

                    best_result = None

                    for k_num in k_num_values:
                        for rho in rho_values:
                            for c in c_values:
                                for theta in theta_values:
                                    for m in m_values:
                                        model_types = ['MLATC-N-node', 'MLATC-T-time', 'MLATC-D-day']
                                        model_type = model_types[m]
                                        alpha = np.ones(3) / 3,
                                        alpha = alpha[0]
                                        start_time = time.time()
                                        tensor_hat, reslut_rmse, initial_a_matrix = mlatc(
                                            dataset_name,
                                            dense_tensor,
                                            sparse_tensor,
                                            alpha,
                                            rho,
                                            c,
                                            theta,
                                            missing_type,
                                            missing_rate,
                                            k_num,
                                            start_time,
                                            model_type,
                                            epsilon_values,
                                            return_initial_a_matrix=True,
                                        )
                                        iter_time = time.time() - start_time
                                        result_mape, result_rmse = calculate_imputation_metrics(dense_tensor, sparse_tensor, tensor_hat)
                                        candidate = {
                                            'dataset_name': dataset_name,
                                            'missing_type': missing_type,
                                            'missing_rate': missing_rate,
                                            'model_name': model_name,
                                            'model_type': model_type,
                                            'rho0': rho,
                                            'k_num': k_num,
                                            'c': c,
                                            'theta': theta,
                                            'epsilon': epsilon_values[-1],
                                            'sparse_tensor_path': str(missing_data_path.relative_to(project_root)),
                                            'random_seed': 1000,
                                            'a_initialization': 'saved_initial_a_matrix',
                                            'initial_a_matrix': initial_a_matrix,
                                        }
                                        if best_result is None or result_rmse < best_result['best_rmse_for_selection']:
                                            candidate['best_rmse_for_selection'] = result_rmse
                                            best_result = candidate
                                            save_current_best_parameters(best_result)
                                        print()

                elif model_name == 'BGCP':
                    for rank in [15, 80]:
                        start = time.time()
                        dim = dense_tensor.shape
                        factor = [0.1 * np.random.randn(dim[k], rank) for k in range(len(dim))]
                        burn_iter = 1000
                        gibbs_iter = 200

                        tensor_hat, res_factor, res_mape, res_rmse = bgcp(dense_tensor, sparse_tensor, factor, burn_iter, gibbs_iter)

                        end = time.time()
                        iter_time = end - start

                        bgcp_bttf_save_result_to_csv(model_name, dataset_name, missing_type, missing_rate, rank, iter_time, res_mape, res_rmse)

                        print('Running time: %d seconds' % (end - start))

                elif model_name == 'BTTF':
                    for rank in [30]:
                        start = time.time()
                        time_lags = np.array([1, 2, 24])
                        init = {"U": 0.1 * np.random.randn(dim1, rank), "V": 0.1 * np.random.randn(dim2, rank),
                                "X": 0.1 * np.random.randn(dim3, rank)}
                        burn_iter = 1000
                        gibbs_iter = 200
                        tensor_hat, u_matrix, v_matrix, x_new, a_matrix, res_mape, res_rmse = bttf(dense_tensor, sparse_tensor, init, rank, time_lags, burn_iter, gibbs_iter)
                        end = time.time()
                        iter_time = end - start
                        bgcp_bttf_save_result_to_csv(model_name, dataset_name, missing_type, missing_rate, rank, iter_time, res_mape, res_rmse)
                        print('Running time: %d seconds' % (end - start))

                elif model_name == 'LRTC_TNN':
                    alpha = np.ones(3) / 3
                    epsilon = 1e-4
                    maxiter = 100

                    for rho in [1e-5, 5e-5, 1e-4]:
                        for theta in [0.05, 0.10, 0.15, 0.20, 0.25, 0.3]:
                            start = time.time()
                            tensor_hat, res_mape, res_rmse = lrtc_tnn(dense_tensor, sparse_tensor, alpha, rho, theta, epsilon, maxiter)
                            end = time.time()
                            print('Running time: %d seconds' % (end - start))
                            iter_time = end - start
                            lrtc_tnn_save_result_to_csv(model_name, dataset_name, missing_type, missing_rate, theta, iter_time, res_mape, res_rmse)

                elif model_name == 'LATC':
                    alpha = np.ones(3) / 3

                    for rho in [1e-5, 1e-4]:
                        for lambda_value in [1 / 10, 1 / 5, 1, 5, 10]:
                            for theta in [5, 10, 15, 20, 25, 30]:
                                start = time.time()
                                time_lags = np.arange(1, 7)
                                alpha = np.ones(3) / 3
                                lambda0 = lambda_value * rho
                                print(lambda_value)
                                print(theta)
                                tensor_hat = latc(dense_tensor, sparse_tensor, time_lags, alpha, rho, lambda0, theta, missing_rate, missing_type, model_name, dataset_name, start)
                                print()

                else:
                    print("Error: model_name is not defined.")
