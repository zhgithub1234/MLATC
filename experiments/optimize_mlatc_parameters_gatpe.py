import optuna
from optuna.samplers import NSGAIISampler, TPESampler
import matplotlib.pyplot as plt
import numpy as np
from hyperopt import hp
from functools import partial
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[1]
datasets_root = project_root / 'datasets'
results_root = project_root / 'results'

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import scipy.io
import os
import time
import csv

from models.mlatc import mlatc
from utils.utils_for_results_save import ten2mat, mat2ten


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


def get_run_time_by_best_parameters(df, dataset_name, missing_type, missing_rate, k_num, rho, c, theta, m):
    """Return cached runtime for one parameter setting, if available."""
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
        iter_time = -1
    elif len(filtered_data) == 1:
        iter_time = filtered_data.iloc[0, -3]
    elif len(filtered_data) > 1:
        iter_time = filtered_data.iloc[-1, -3]

    return iter_time

def objective(trial):
    k_num = trial.suggest_categorical('k_num', [3, 6, 9, 12])
    theta = trial.suggest_categorical('theta', [5, 10, 15, 20, 25, 30])
    m = trial.suggest_categorical('m', ['MLATC-N-node', 'MLATC-T-time', 'MLATC-D-day'])

    rho = trial.suggest_categorical('rho', [1e-5, 5e-5, 1e-4])
    c = trial.suggest_categorical('c', [0.1, 0.2, 1, 5, 10])

    loss = get_imputation_error(
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

    return loss

def record_loss(study, trial):
    loss_history.append(trial.value)
    history['k_num'].append(trial.params['k_num'])
    history['rho'].append(trial.params['rho'])
    history['c'].append(trial.params['c'])
    history['theta'].append(trial.params['theta'])
    history['m'].append(trial.params['m'])
    history['loss'].append(trial.value)

    it_ietr_time = get_run_time_by_best_parameters(calculated_resluts, dataset_name, missing_type, missing_rate,
                                                   trial.params['k_num'], trial.params['rho'], trial.params['c'], trial.params['theta'], trial.params['m'])
    history['iter_time'].append(it_ietr_time)

def save_history_to_csv(history, save_file, filename):
    """Save optimization history to a CSV file."""

    df = pd.DataFrame(history)

    if save_file is not None:
        full_path = os.path.join(save_file, filename)
        os.makedirs(save_file, exist_ok=True)
    else:
        full_path = filename

    df.to_csv(full_path, index=False, encoding='utf-8')

model_name = 'MLATC'
dataset_name = 'Guangzhou'
optimal_models = ['GA-TPE']
missing_types = ['RM']
missing_rates = [0.3]

for iter_num in range(1):
    for missing_type in missing_types:
        for missing_rate in missing_rates:
            for optimal_model in optimal_models:
                csv_path = str(results_root / 'model_results' / dataset_name / model_name / 'output.csv')
                calculated_resluts = pd.read_csv(csv_path)

                loss_history = []
                max_evals = 100
                history = {
                    'k_num': [],
                    'rho': [],
                    'c': [],
                    'theta': [],
                    'm': [],
                    'loss': [],
                    'iter_time': [],
                }
                switch_rate = 0.6

                if optimal_model == 'GA-TPE':
                    study_ga = optuna.create_study(
                        direction="minimize",
                        sampler=NSGAIISampler(
                            crossover_prob=0.8,
                            mutation_prob=0.1,
                            swapping_prob=0.4,
                        )
                    )

                    ga_trials = int(max_evals * switch_rate)
                    study_ga.optimize(objective, n_trials=ga_trials, callbacks=[record_loss])

                    study_tpe = optuna.create_study(
                        direction="minimize",
                        sampler=TPESampler(
                            n_startup_trials=0,
                            multivariate=True
                        )
                    )
                    for trial in study_ga.trials:
                        study_tpe.add_trial(trial)
                    study_tpe.optimize(objective, n_trials=max_evals - ga_trials, callbacks=[record_loss])
                else:
                    raise ValueError(f"Unknown optimal_model: {optimal_model}")

                print("\n=== Final Results ===")
                print(f"Best value: {study_tpe.best_value:.6f}")
                print("Best params:")
                for k, v in study_tpe.best_params.items():
                    print(f"  {k}: {v}")

                plt.figure(figsize=(12, 6))
                plt.plot(np.arange(len(loss_history)), loss_history, 'b-', label='Loss', alpha=0.7)
                plt.scatter(
                    np.argmin(loss_history), np.min(loss_history),
                    color='red', s=100,
                    label=f'Best (Iter={np.argmin(loss_history) + 1}, Loss={np.min(loss_history):.4f})'
                )

                plt.axvline(x=max_evals * switch_rate, color='r', linestyle='--', alpha=0.5, label='GA → TPE')
                plt.xlabel("Iteration")
                plt.ylabel("Loss")
                plt.title("Optimization Progress " + optimal_model)
                plt.legend()
                plt.grid(alpha=0.3)

                plt.tight_layout()

                best_k_num = study_tpe.best_params['k_num']
                best_rho = study_tpe.best_params['rho']
                best_c = study_tpe.best_params['c']
                best_theta = study_tpe.best_params['theta']
                best_m = study_tpe.best_params['m']
                best_loss = study_tpe.best_value

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

                png_save_path = str(results_root / "optimal_parameters" / model_name / dataset_name / missing_type / str(missing_rate) / optimal_model) + os.sep
                save_name = f"{max_evals}_{best_k_num}_{best_rho}_{best_c}_{best_theta}_{best_m}_{best_loss}_{iter_num}.png"
                if not os.path.exists(os.path.dirname(png_save_path)):
                    os.makedirs(os.path.dirname(png_save_path))
                plt.savefig(png_save_path + save_name, dpi=300)
                plt.close()

                save_history_to_csv(history, png_save_path, save_name.replace(".png", ".csv"))
