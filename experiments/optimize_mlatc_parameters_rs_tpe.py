"""Optimize MLATC parameters with RS and TPE."""

import numpy as np
import pandas as pd
import scipy.io
import os
import time
import random
import csv
from functools import partial
import matplotlib.pyplot as plt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[1]
datasets_root = project_root / 'datasets'
results_root = project_root / 'results'

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


from models.mlatc import mlatc
from utils.utils_for_results_save import ten2mat, mat2ten

iteration_counter = 0
result_columns = ['Missing_type', 'Model_type', 'Missing_rate', 'Rho0', 'K_num', 'c', 'theta', 'Epsilon', 'Iter_num', 'Iter_time', 'MAPE', 'RMSE']

def load_calculated_results(csv_path):
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return pd.DataFrame(columns=result_columns)

def get_missing_data(dataset_name, missing_type, missing_rate):

    datasets_paths = {
        'Guangzhou': str(datasets_root / 'Guangzhou-data-set' / 'tensor.mat'),
        'Hangzhou': str(datasets_root / 'Hangzhou-data-set' / 'tensor.mat'),
        'Seattle': str(datasets_root / 'Seattle-data-set' / 'tensor.npz')
    }
    data_path = datasets_paths[dataset_name]
    if 'Guangzhou' in dataset_name or 'Hangzhou' in dataset_name:
        dense_tensor = scipy.io.loadmat(data_path)['tensor'].transpose(0, 2, 1)
    elif 'Seattle' in dataset_name:
        dense_tensor = np.load(data_path)['arr_0'].transpose(0, 2, 1)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dim1, dim2, dim3 = dense_tensor.shape

    missing_data_path = datasets_root / 'generated_missing_data' / dataset_name / missing_type / f'{missing_rate}.npy'
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
            sparse_tensor = mat2ten(ten2mat(dense_tensor, 0) * np.round(vec + 0.5 - missing_rate)[None, :],
                                    np.array([dim1, dim2, dim3]), 0)

        else:
            raise ValueError(f"Unknown missing type: {missing_type}")

    return dense_tensor, sparse_tensor

def get_imputation_error(df, dataset_name, missing_type, missing_rate, k_num, rho, c, theta, m):
    """Return cached RMSE for one parameter setting, if available."""
    condition = (
            (df['Missing_type'] == missing_type) &
            (df['Missing_rate'] == missing_rate) &
            (df['Rho0'] == rho) &
            (df['K_num'] == k_num) &
            (df['c'] == c) &
            (df['theta'] == theta) &
            (df['Model_type'] == m)
    )

    filtered_data = df.loc[condition]

    if len(filtered_data) == 0:
        dense_tensor, sparse_tensor = get_missing_data(dataset_name, missing_type, missing_rate)
        alpha = np.ones(3) / 3
        start_time = time.time()
        model_type = m
        epsilon_values = [1e-4]
        _, reslut_rmse = mlatc(dataset_name, dense_tensor, sparse_tensor, alpha, rho, c, theta,
                               missing_type, missing_rate, k_num, start_time, model_type,
                               epsilon_values)
        df.loc[len(df)] = [
            missing_type,
            model_type,
            missing_rate,
            rho,
            k_num,
            c,
            theta,
            epsilon_values[-1],
            np.nan,
            np.nan,
            np.nan,
            round(reslut_rmse, 2),
        ]
        return reslut_rmse
    elif len(filtered_data) > 1:
        return filtered_data.iloc[-1, -1]
    else:
        return filtered_data.iloc[0, -1]

def model(params, space_dictionary, optimal_model, dataset_name, missing_type, missing_rate, calculated_resluts, history,
          sampling_mode='discrete'):
    global iteration_counter

    k_num = space_dictionary['k_num'][params['k_num']]
    theta = space_dictionary['theta'][params['theta']]
    m = space_dictionary['m'][params['m']]

    if sampling_mode == 'continuous':
        rho = params['rho']
        c = params['c']
    elif sampling_mode == 'discrete':
        rho = space_dictionary['rho'][params['rho']]
        c = space_dictionary['c'][params['c']]
    else:
        raise ValueError(f"Unknown sampling mode: {sampling_mode}")

    accuracy = get_imputation_error(
        calculated_resluts,
        dataset_name=dataset_name,
        missing_type=missing_type,
        missing_rate=missing_rate,
        k_num=k_num,
        rho=rho,
        c=c,
        theta=theta,
        m=m
    )

    history['iteration'].append(iteration_counter)
    history['k_num'].append(k_num)
    history['rho'].append(rho)
    history['c'].append(c)
    history['theta'].append(theta)
    history['m'].append(m)
    history['accuracy'].append(accuracy)
    history['loss'].append(accuracy)

    print(f"Iteration {iteration_counter}: k_num={k_num}, rho={rho}, c={c}, theta={theta}, m={m}, accuracy={accuracy}")
    print()
    iteration_counter += 1

    return {'loss': accuracy, 'status': STATUS_OK}


def random_sample_params(space_dictionary, sampling_mode):
    params = {
        'k_num': random.choice(range(4)),
        'theta': random.choice(range(6)),
        'm': random.choice(range(3)),
    }

    if sampling_mode == 'continuous':
        params['rho'] = round(random.uniform(1e-5, 1e-3) / 1e-6) * 1e-6
        params['c'] = round(random.uniform(0.1, 10) / 1e-2) * 1e-2
    elif sampling_mode == 'discrete':
        params['rho'] = random.choice(range(len(space_dictionary['rho'])))
        params['c'] = random.choice(range(len(space_dictionary['c'])))
    else:
        raise ValueError(f"Unknown sampling mode: {sampling_mode}")

    return params

def random_search(max_evals, space_dictionary, optimal_model, dataset_name, missing_type, missing_rate, calculated_resluts, history,
                  sampling_mode):
    best_loss = float('inf')
    best_params = None

    for i in range(max_evals):
        params = random_sample_params(space_dictionary, sampling_mode)
        result = model(params, space_dictionary, optimal_model, dataset_name, missing_type, missing_rate,
                       calculated_resluts, history, sampling_mode)

        if result['loss'] < best_loss:
            best_loss = result['loss']
            best_params = params
            print(f"New best found at iteration {i}: loss={best_loss:.4f}")

    return best_params, best_loss

def plot_loss_curve(history, save_file, save_name):
    """Plot the optimization loss curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(history['iteration'], history['loss'], 'b-o', markersize=4)
    plt.xlabel('Iteration')
    plt.ylabel('Loss (Accuracy)')
    plt.title('Loss Curve During Hyperparameter Optimization')
    plt.grid(True, alpha=0.3)

    best_idx = np.argmin(history['loss'])
    plt.scatter(history['iteration'][best_idx], history['loss'][best_idx],
                c='r', s=100, label=f'Best (Iteration {best_idx})')
    plt.legend()

    plt.tight_layout()

    if not os.path.exists(os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file))
    plt.savefig(save_file + save_name, dpi=300)
    plt.close()


def save_history_to_csv(history, save_file, filename):
    """Save optimization history to a CSV file."""

    df = pd.DataFrame(history)

    if save_file is not None:
        full_path = os.path.join(save_file, filename)
        os.makedirs(save_file, exist_ok=True)
    else:
        full_path = filename

    df.to_csv(full_path, index=False, encoding='utf-8')


def get_optimal_parameters():
    global iteration_counter

    model_name = 'MLATC'
    dataset_name = 'Guangzhou'

    missing_types = ['RM']
    missing_rates = [0.3]
    max_evals = 100

    space = {
        'k_num': hp.choice('k_num', range(4)),
        'rho': hp.choice('rho', range(3)),
        'c': hp.choice('c', range(5)),
        'theta': hp.choice('theta', range(6)),
        'm': hp.choice('m', range(3)),
    }

    space_dictionary = {
        'k_num': [3, 6, 9, 12],
        'rho': [1e-5, 5e-5, 1e-4],
        'c': [0.1, 0.2, 1, 5, 10],
        'theta': [5, 10, 15, 20, 25, 30],
        'm': ['MLATC-N-node', 'MLATC-T-time', 'MLATC-D-day']
    }

    search_configs = [
        ('TPE', 'discrete'),
        ('RS-continuous', 'continuous'),
        ('RS-discrete', 'discrete'),
    ]

    for missing_type in missing_types:
        for missing_rate in missing_rates:
            csv_path = str(results_root / 'model_results' / dataset_name / model_name / 'output.csv')
            calculated_resluts = load_calculated_results(csv_path)

            for optimal_model, sampling_mode in search_configs:
                for iter_num in range(1):
                    iteration_counter = 0

                    history = {
                        'iteration': [],
                        'k_num': [],
                        'rho': [],
                        'c': [],
                        'theta': [],
                        'm': [],
                        'accuracy': [],
                        'loss': []
                    }

                    if optimal_model == 'TPE':
                        trials = Trials()
                        model_with_args = partial(
                            model,
                            space_dictionary=space_dictionary,
                            optimal_model=optimal_model,
                            dataset_name=dataset_name,
                            missing_type=missing_type,
                            missing_rate=missing_rate,
                            calculated_resluts=calculated_resluts,
                            history=history,
                            sampling_mode=sampling_mode,
                        )
                        best = fmin(fn=model_with_args, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
                        best_loss = trials.best_trial['result']['loss']
                    elif optimal_model.startswith('RS'):
                        best, best_loss = random_search(
                            max_evals=max_evals,
                            space_dictionary=space_dictionary,
                            optimal_model=optimal_model,
                            dataset_name=dataset_name,
                            missing_type=missing_type,
                            missing_rate=missing_rate,
                            calculated_resluts=calculated_resluts,
                            history=history,
                            sampling_mode=sampling_mode,
                        )
                    else:
                        raise ValueError(f"Unknown optimal model: {optimal_model}")

                    best_k_num = space_dictionary['k_num'][best['k_num']]
                    best_theta = space_dictionary['theta'][best['theta']]
                    best_m = space_dictionary['m'][best['m']]
                    best_loss = round(best_loss, 2)

                    if sampling_mode == 'discrete':
                        best_rho = space_dictionary['rho'][best['rho']]
                        best_c = space_dictionary['c'][best['c']]
                    else:
                        best_rho = round(best['rho'], 6)
                        best_c = round(best['c'], 2)

                    csv_file = str(results_root / 'optimal_parameters' / model_name) + os.sep
                    if not os.path.exists(os.path.dirname(csv_file)):
                        os.makedirs(os.path.dirname(csv_file))
                    with open(csv_file + 'output.csv', 'a', newline='') as f:
                        writer = csv.writer(f)
                        if os.path.getsize(csv_file + 'output.csv') == 0:
                            writer.writerow(
                                ['Model_name', 'Dataset_name', 'Missing_type', 'Missing_rate', 'Optimal_model', 'Max_evals',
                                 'best_k_num', 'best_rho', 'best_c', 'best_theta', 'best_m', 'best_loss'])
                        writer.writerow(
                            [model_name, dataset_name, missing_type, missing_rate, optimal_model, max_evals, best_k_num,
                             best_rho, best_c, best_theta, best_m, best_loss])

                    print(f"Best parameters: k_num={best_k_num}, rho={best_rho}, c={best_c}, theta={best_theta}, m={best_m}")
                    png_save_path = str(results_root / "optimal_parameters" / model_name / dataset_name / missing_type / str(missing_rate) / optimal_model) + os.sep
                    save_name = f"{max_evals}_{best_k_num}_{best_rho}_{best_c}_{best_theta}_{best_m}_{best_loss}_{iter_num}.png"
                    plot_loss_curve(history, png_save_path, save_name)
                    save_history_to_csv(history, png_save_path, save_name.replace(".png", ".csv"))


if __name__ == '__main__':
    for i in range(1):
        get_optimal_parameters()