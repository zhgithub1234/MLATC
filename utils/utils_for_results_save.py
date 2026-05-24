"""Utilities for tensor reshaping and baseline result persistence."""

import csv
import os

import numpy as np


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_root = os.path.join(project_root, 'results')


def ten2mat(tensor, mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order='F')


def mat2ten(mat, tensor_size, mode):
    index = [mode]
    for i in range(tensor_size.shape[0]):
        if i != mode:
            index.append(i)
    return np.moveaxis(np.reshape(mat, tensor_size[index].tolist(), order='F'), 0, mode)


def bgcp_bttf_save_result_to_csv(model_name, dataset_name, missing_type, missing_rate, rank, iter_time, reslut_mape, reslut_rmse):
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


def lrtc_tnn_save_result_to_csv(model_name, dataset_name, missing_type, missing_rate, theta, iter_time, reslut_mape, reslut_rmse):
    csv_file_path = os.path.join(results_root, 'model_results', dataset_name, model_name)
    if not os.path.exists(csv_file_path):
        os.makedirs(csv_file_path)

    csv_path = os.path.join(csv_file_path, 'output.csv')
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(['Missing_type', 'Missing_rate', 'Theta', 'Iter_time', 'MAPE', 'RMSE'])

        reslut_mape = round(reslut_mape * 100, 2)
        reslut_rmse = round(reslut_rmse, 2)
        iter_time = round(iter_time, 2)
        writer.writerow([missing_type, missing_rate, theta, iter_time, reslut_mape, reslut_rmse])
