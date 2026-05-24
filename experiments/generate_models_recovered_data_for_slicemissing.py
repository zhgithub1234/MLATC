"""Generate recovered tensors for Figure 5 slice-missing visualizations."""

import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io
import time
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
datasets_root = project_root / 'datasets'
results_root = project_root / 'results'

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from baseline_models.BGCP.BGCP import bgcp
from baseline_models.BTTF.BTTF import bttf
from baseline_models.LRTC_TNN.LRTC_TNN import lrtc as lrtc_tnn
from baseline_models.LATC.LATC import latc

from models.mlatc import mlatc

# missing cases settings
missing_types = ['NodeM', 'TimeM', 'DayM']
selected_paras_idxs = [0, 1, 2]
for selected_paras_idx in selected_paras_idxs:
    missing_type = missing_types[selected_paras_idx]
    tensor_path = str(datasets_root / 'Guangzhou-data-set' / 'tensor.mat')
    dense_tensor = scipy.io.loadmat(tensor_path)['tensor'].transpose(0, 2, 1)
    dim = dense_tensor.shape
    sparse_data_path = str(results_root / 'slice_missing_data' / 'Guangzhou' / (missing_type + '_0.3.npz'))
    data = np.load(sparse_data_path)
    sparse_tensor = data['sparse_tensor']

    mode_names = ['BGCP', 'BTTF', 'LRTC_TNN', 'LATC', 'MLATC']
    missing_rate = 0.3
    dataset_name = 'Guangzhou'
    # Run LATC and baseline methods.
    for mode_name in mode_names:
        if mode_name == 'BGCP':
            rank = 80
            burn_iter = 1000
            gibbs_iter = 200
            factor = [0.1 * np.random.randn(dim[k], rank) for k in range(len(dim))]
            tensor_hat, res_factor, res_mape, res_rmse = bgcp(dense_tensor, sparse_tensor, factor, burn_iter, gibbs_iter)

        elif mode_name == 'BTTF':
            start = time.time()
            dim1, dim2, dim3 = sparse_tensor.shape
            rank = 30
            time_lags = np.array([1, 2, 24])
            init = {"U": 0.1 * np.random.randn(dim1, rank), "V": 0.1 * np.random.randn(dim2, rank),
                    "X": 0.1 * np.random.randn(dim3, rank)}
            burn_iter = 10
            gibbs_iter = 2
            tensor_hat, u_matrix, v_matrix, x_new, a_matrix, res_mape, res_rmse = bttf(dense_tensor, sparse_tensor, init, rank, time_lags, burn_iter, gibbs_iter)

        elif mode_name == 'LRTC_TNN':

            alpha = np.ones(3) / 3
            rho = 1e-5
            if missing_type == 'NodeM':
                theta = 0.1
            elif missing_type == 'TimeM':
                theta = 0.15
            elif missing_type == 'DayM':
                theta = 0.1

            epsilon = 1e-4
            maxiter = 100
            tensor_hat, res_mape, res_rmse = lrtc_tnn(dense_tensor, sparse_tensor, alpha, rho, theta, epsilon, maxiter)

        elif mode_name == 'LATC':
            time_lags = np.arange(1, 7)
            alpha = np.ones(3) / 3
            rho = 1e-5
            if missing_type == 'NodeM':
                c = 0.1
                theta = 5
            elif missing_type == 'TimeM':
                c = 10
                theta = 30
            elif missing_type == 'DayM':
                c = 10
                theta = 30

            lambda0 = c * rho
            start = time.time()

            tensor_hat = latc(dense_tensor, sparse_tensor, time_lags, alpha, rho, lambda0, theta, missing_rate, missing_type, mode_name, dataset_name, start)

        elif mode_name == 'MLATC':
            bestparas_sm1 = {
                'MLATC': {
                    'm_value': 0,
                    'rho': 5e-5,
                    'k_num': 6,
                    'c': 0.2,
                    'theta': 5,
                }
            }
            bestparas_sm2 = {
                'MLATC': {
                    'm_value': 1,
                    'rho': 1e-4,
                    'k_num': 6,
                    'c': 10,
                    'theta': 10,
                }
            }
            bestparas_sm3 = {
                'MLATC': {
                    'm_value': 2,
                    'rho': 5e-5,
                    'k_num': 3,
                    'c': 5,
                    'theta': 15,
                }
            }

            epsilon_values = [1e-4]
            bestparas = [bestparas_sm1, bestparas_sm2, bestparas_sm3]
            selected_paras = bestparas[selected_paras_idx][mode_name]
            c = selected_paras['c']
            theta = selected_paras['theta']
            m = selected_paras['m_value']
            rho = selected_paras['rho']
            k_num = selected_paras['k_num']
            model_types = ['MLATC-N-node', 'MLATC-T-time', 'MLATC-D-day']
            model_type = model_types[selected_paras_idx]
            alpha = np.ones(3) / 3,
            alpha = alpha[0]
            start_time = time.time()

            tensor_hat, reslut_rmse = mlatc(dataset_name, dense_tensor, sparse_tensor, alpha, rho, c, theta, missing_type, missing_rate, k_num, start_time, model_type, epsilon_values)


        root_path = str(results_root / 'slice_recovered_data' / dataset_name / missing_type)
        if not os.path.exists(root_path):
            os.makedirs(root_path)

        npz_file_name = f"{mode_name}_{missing_type}_{missing_rate}.npz"
        npz_file_path = os.path.join(root_path, npz_file_name)
        # npz save
        np.savez(npz_file_path, tensor_hat=tensor_hat)



