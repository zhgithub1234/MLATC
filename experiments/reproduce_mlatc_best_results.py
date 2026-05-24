import sys
import time
from pathlib import Path

import numpy as np
import scipy.io


project_root = Path(__file__).resolve().parents[1]
datasets_root = project_root / 'datasets'
results_root = project_root / 'results'
legacy_datasets_root = project_root.parent / 'LATC-improved' / 'datasets'

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models.mlatc import compute_mape, compute_rmse, mlatc


dataset_names = ['Guangzhou', 'Hangzhou', 'Seattle']
alpha = np.ones(3) / 3


def load_dense_tensor(dataset_name):
    if dataset_name == 'Guangzhou':
        data_path = datasets_root / 'Guangzhou-data-set' / 'tensor.mat'
        return scipy.io.loadmat(str(data_path))['tensor'].transpose(0, 2, 1)
    if dataset_name == 'Hangzhou':
        data_path = datasets_root / 'Hangzhou-data-set' / 'tensor.mat'
        return scipy.io.loadmat(str(data_path))['tensor'].transpose(0, 2, 1)
    if dataset_name == 'Seattle':
        data_path = datasets_root / 'Seattle-data-set' / 'tensor.npz'
        if not data_path.exists():
            data_path = legacy_datasets_root / 'Seattle-data-set' / 'tensor.npz'
        return np.load(str(data_path))['arr_0'].transpose(0, 2, 1)
    raise ValueError('Unknown dataset: {}'.format(dataset_name))


def load_parameter_packages(dataset_name):
    parameter_dir = results_root / 'model_results' / dataset_name / 'MLATC' / 'best_parameters'
    if not parameter_dir.exists():
        print('Skipping {} because best parameter directory is missing: {}'.format(dataset_name, parameter_dir))
        return []

    parameter_paths = sorted(parameter_dir.glob('*.npy'))
    if not parameter_paths:
        print('Skipping {} because no best parameter npy files were found in {}'.format(dataset_name, parameter_dir))
        return []

    return parameter_paths


def load_best_parameters(parameter_path):
    parameters = np.load(str(parameter_path), allow_pickle=True).item()
    required_keys = [
        'dataset_name',
        'missing_type',
        'missing_rate',
        'model_type',
        'rho0',
        'k_num',
        'c',
        'theta',
        'epsilon',
        'sparse_tensor_path',
        'initial_a_matrix',
    ]
    missing_keys = [key for key in required_keys if key not in parameters]
    if missing_keys:
        raise ValueError('{} is missing keys: {}'.format(parameter_path, ', '.join(missing_keys)))
    return parameters


def resolve_sparse_tensor_path(parameters):
    saved_path = Path(str(parameters['sparse_tensor_path']))
    if saved_path.exists():
        return saved_path

    if not saved_path.is_absolute():
        project_path = project_root / saved_path
        if project_path.exists():
            return project_path

    current_path = (
        datasets_root
        / 'generated_missing_data'
        / parameters['dataset_name']
        / parameters['missing_type']
        / '{}.npy'.format(parameters['missing_rate'])
    )
    if current_path.exists():
        return current_path

    raise FileNotFoundError(
        'Missing sparse tensor file. Saved path: {}. Current project path: {}'.format(
            saved_path,
            current_path,
        )
    )


def reproduce_parameters(parameters, dense_tensor):
    sparse_tensor = np.load(str(resolve_sparse_tensor_path(parameters)))
    initial_a_matrix = parameters['initial_a_matrix']

    start_time = time.time()
    tensor_hat, _ = mlatc(
        parameters['dataset_name'],
        dense_tensor,
        sparse_tensor,
        alpha,
        float(parameters['rho0']),
        float(parameters['c']),
        int(parameters['theta']),
        parameters['missing_type'],
        float(parameters['missing_rate']),
        int(parameters['k_num']),
        start_time,
        parameters['model_type'],
        [float(parameters['epsilon'])],
        save_outputs=False,
        initial_a_matrix=initial_a_matrix,
    )

    pos_test = np.where((dense_tensor != 0) & (sparse_tensor == 0))
    dense_test = dense_tensor[pos_test]
    result_mape = round(compute_mape(dense_test, tensor_hat[pos_test]) * 100, 2)
    result_rmse = round(compute_rmse(dense_test, tensor_hat[pos_test]), 2)
    return result_mape, result_rmse


def main():
    for dataset_name in dataset_names:
        parameter_paths = load_parameter_packages(dataset_name)
        if not parameter_paths:
            continue

        dense_tensor = load_dense_tensor(dataset_name)
        print('\nDataset: {}'.format(dataset_name))
        for index, parameter_path in enumerate(parameter_paths, 1):
            parameters = load_best_parameters(parameter_path)
            print('Reproducing {}/{}: {}, rate {}, {}'.format(
                index,
                len(parameter_paths),
                parameters['missing_type'],
                parameters['missing_rate'],
                parameters['model_type'],
            ))
            reproduce_parameters(parameters, dense_tensor)

if __name__ == '__main__':
    main()
