"""Mode-wise unfolding and Pearson correlation plots."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from scipy import sparse
import scipy.io
from matplotlib.gridspec import GridSpec
import os
from pathlib import Path


project_root = Path(__file__).resolve().parents[1]
default_save_dir = str(project_root / 'results' / 'figure_results' / 'figure1_mode_vis')

def compute_node_correlation(mode_data, k=6):
    """Build a top-k Pearson correlation mask for one unfolded mode."""

    correlation_matrix = np.corrcoef(mode_data)
    adj_matrix = np.zeros_like(correlation_matrix)

    for i in range(correlation_matrix.shape[0]):
        corr_row = correlation_matrix[i, :]
        corr_row[i] = -np.inf
        top_k_indices = np.argpartition(corr_row, -k)[-k:]
        adj_matrix[i, top_k_indices] = 1

    # Mark the diagonal separately for visualization.
    for i in range(correlation_matrix.shape[0]):
        adj_matrix[i, i] = 2

    return adj_matrix


def ten2mat(tensor, mode):
    """Unfold a tensor along the selected mode."""
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order='F')


def mat2ten(mat, tensor_size, mode):
    """Fold a mode-unfolded matrix back to tensor form."""
    index = list()
    index.append(mode)
    for i in range(tensor_size.shape[0]):
        if i != mode:
            index.append(i)
    return np.moveaxis(np.reshape(mat, tensor_size[index].tolist(), order='F'), 0, mode)


def save_mode0_pair_visualization(dense_tensor, mode_idx, mode_data_raw, mode_corr_matrix, dataset_name, save_dir=default_save_dir):
    """Save the mode-0 unfolding and correlation panels."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fontsize_set = 26

    fig = plt.figure(figsize=(15, 3), dpi=50)
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = fontsize_set
    plt.rcParams['font.weight'] = 'bold'

    time_intervals_of_day = dense_tensor.shape[1]
    locations_num = dense_tensor.shape[0]
    days_num = dense_tensor.shape[2]

    mode_0_day_length = 3
    total_time_intervals = mode_0_day_length * time_intervals_of_day

    gs = GridSpec(1, 2, figure=fig, width_ratios=[6, 1], wspace=0.2)

    # Plot the unfolded tensor values.
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_position([0.08, 0.15, 0.65, 0.75])

    im1 = ax1.imshow(
        mode_data_raw[:, :total_time_intervals],
        aspect='auto',
        cmap='jet',
        vmin=0,
        vmax=60
    )

    ax1.set_ylabel('Locations', fontsize=fontsize_set, fontweight='bold', fontfamily='Times New Roman')
    ax1.set_xlabel('Days × time intervals', fontsize=fontsize_set, fontweight='bold', fontfamily='Times New Roman')

    # Label the displayed day blocks.
    total_days = mode_0_day_length
    date_start = '2016-08-01'
    base_date = np.datetime64(date_start)
    dates = [base_date + np.timedelta64(i, 'D') for i in range(total_days)]

    num_ticks = min(5, total_days)
    step = max(1, total_days // num_ticks)

    date_ticks = list(range(0, total_time_intervals, time_intervals_of_day * step))
    date_ticks = [x + time_intervals_of_day / 2 for x in date_ticks]
    date_labels = [str(dates[i]) for i in range(0, total_days, step)]

    date_ticks = date_ticks[:num_ticks]
    date_labels = date_labels[:num_ticks]

    ax1.set_xticks(date_ticks)
    ax1.set_xticklabels(date_labels, rotation=0, ha='center')

    cbar1 = plt.colorbar(im1, ax=ax1, orientation='vertical',
                         pad=0.03,
                         shrink=0.8,
                         aspect=35,
                         fraction=0.03)
    cbar1.ax.tick_params(labelsize=fontsize_set)

    # Plot the top-k Pearson correlation mask.
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_position([0.6, 0.25, 0.2, 0.75])

    im_mode0 = ax2.imshow(
        mode_corr_matrix,
        cmap='YlOrRd',
        vmin=0,
        vmax=2,
        aspect='auto'
    )

    plt.subplots_adjust(left=0.06, right=0.98, bottom=0.15, top=0.95, wspace=0.12)

    save_path_png = os.path.join(save_dir, f'mode{mode_idx}_visualization_{dataset_name}.png')
    save_path_eps = os.path.join(save_dir, f'mode{mode_idx}_visualization_{dataset_name}.eps')
    plt.savefig(save_path_png, dpi=fontsize_set, bbox_inches='tight')
    plt.savefig(save_path_eps, dpi=fontsize_set, bbox_inches='tight')
    plt.close()
    print(f"Saved Mode {mode_idx} visualization to {save_path_png} and {save_path_eps}")

def save_mode1_pair_visualization(dense_tensor, mode_idx, mode_data_raw, mode_corr_matrix, dataset_name, save_dir=default_save_dir):
    """Save the mode-1 unfolding and correlation panels."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fontsize_set = 26

    fig = plt.figure(figsize=(15, 3), dpi=50)
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = fontsize_set
    plt.rcParams['font.weight'] = 'bold'

    time_intervals_of_day = dense_tensor.shape[1]
    locations_num = dense_tensor.shape[0]
    days_num = dense_tensor.shape[2]
    gs = GridSpec(1, 2, figure=fig, width_ratios=[6, 1], wspace=0.2)
    mode_0_day_length = 3

    mode_2_time_length = 3
    mode_2_total_time_intervals = mode_2_time_length * time_intervals_of_day

    # Plot the unfolded tensor values.
    ax2 = fig.add_subplot(gs[0, 0])
    im2 = ax2.imshow(
        mode_data_raw[:, :mode_2_total_time_intervals],
        aspect='auto',
        cmap='jet',
        vmin=0,
        vmax=60
    )

    ax2.set_ylabel('Days', fontsize=fontsize_set, fontweight='bold', fontfamily='Times New Roman')
    ax2.set_xlabel('Locations × time intervals', fontsize=fontsize_set, fontweight='bold', fontfamily='Times New Roman')

    # Label the displayed location blocks.
    total_locations = mode_0_day_length
    locations = ["Location-" + str(i + 1) for i in range(total_locations)]

    num_ticks = min(5, total_locations)
    step = max(1, total_locations // num_ticks)

    date_ticks = list(range(0, mode_2_total_time_intervals, time_intervals_of_day * step))
    date_ticks = [x + time_intervals_of_day / 2 for x in date_ticks]
    date_labels = [str(locations[i]) for i in range(0, total_locations, step)]

    date_ticks = date_ticks[:num_ticks]
    date_labels = date_labels[:num_ticks]

    ax2.set_xticks(date_ticks)
    ax2.set_xticklabels(date_labels, rotation=0, ha='center')

    cbar2 = plt.colorbar(im2, ax=ax2, orientation='vertical',
                         pad=0.03,
                         shrink=0.8,
                         aspect=35,
                         fraction=0.03)
    cbar2.ax.tick_params(labelsize=fontsize_set)


    # Plot the top-k Pearson correlation mask.
    ax2 = fig.add_subplot(gs[0, 1])
    im_mode0 = ax2.imshow(
        mode_corr_matrix,
        cmap='YlOrRd',
        vmin=0,
        vmax=2,
        aspect='auto'
    )
    # Highlight weekly periodic offsets.
    step = 7
    n_rows, n_cols = mode_corr_matrix.shape
    for offset in range(0, n_cols, step):
        ax2.plot(
            [0, n_cols - 1 - offset],
            [0 + offset, n_cols - 1],
            color='blue', linestyle='--', alpha=0.6, linewidth=1.5
        )
        ax2.plot(
            [0 + offset, n_cols - 1],
            [0, n_cols - 1 - offset],
            color='blue', linestyle='--', alpha=0.6, linewidth=1.5
        )

    ax2.set_position([0.5, 0.25, 0.2, 0.75])

    plt.subplots_adjust(left=0.06, right=0.98, bottom=0.15, top=0.95, wspace=0.12)

    save_path_png = os.path.join(save_dir, f'mode{mode_idx}_visualization_{dataset_name}.png')
    save_path_eps = os.path.join(save_dir, f'mode{mode_idx}_visualization_{dataset_name}.eps')
    plt.savefig(save_path_png, dpi=fontsize_set, bbox_inches='tight')
    plt.savefig(save_path_eps, dpi=fontsize_set, bbox_inches='tight')
    plt.close()
    print(f"Saved Mode {mode_idx} visualization to {save_path_png} and {save_path_eps}")

def save_mode2_pair_visualization(dense_tensor, mode_idx, mode_data_raw, mode_corr_matrix, dataset_name, save_dir=default_save_dir):
    """Save the mode-2 unfolding and correlation panels."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fontsize_set = 26

    fig = plt.figure(figsize=(15, 3), dpi=50)
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = fontsize_set
    plt.rcParams['font.weight'] = 'bold'

    time_intervals_of_day = dense_tensor.shape[1]
    locations_num = dense_tensor.shape[0]
    days_num = dense_tensor.shape[2]
    gs = GridSpec(1, 2, figure=fig, width_ratios=[6, 1], wspace=0.2)
    mode_0_day_length = 3

    mode_3_time_length = 3
    mode_3_total_time_intervals = mode_3_time_length * days_num

    # Plot the unfolded tensor values.
    ax3 = fig.add_subplot(gs[0, 0])
    im3 = ax3.imshow(
        mode_data_raw[:, :mode_3_total_time_intervals],
        aspect='auto',
        cmap='jet',
        vmin=0,
        vmax=60
    )

    ax3.set_ylabel('Time intervals', fontsize=fontsize_set, fontweight='bold', fontfamily='Times New Roman')
    ax3.set_xlabel('Locations × days', fontsize=fontsize_set, fontweight='bold', fontfamily='Times New Roman')

    # Label the displayed location blocks.
    total_locations = mode_0_day_length
    locations = ["Location-" + str(i + 1) for i in range(total_locations)]

    num_ticks = min(5, total_locations)
    step = max(1, total_locations // num_ticks)

    date_ticks = list(range(0, mode_3_total_time_intervals, days_num * step))
    date_ticks = [x + days_num / 2 for x in date_ticks]
    date_labels = [str(locations[i]) for i in range(0, total_locations, step)]

    date_ticks = date_ticks[:num_ticks]
    date_labels = date_labels[:num_ticks]

    ax3.set_xticks(date_ticks)
    ax3.set_xticklabels(date_labels, rotation=0, ha='center')

    cbar3 = plt.colorbar(im3, ax=ax3, orientation='vertical',
                         pad=0.03,
                         shrink=0.8,
                         aspect=35,
                         fraction=0.03)
    cbar3.ax.tick_params(labelsize=fontsize_set)

    # Plot the top-k Pearson correlation mask.
    ax3 = fig.add_subplot(gs[0, 1])
    im_mode0 = ax3.imshow(
        mode_corr_matrix,
        cmap='YlOrRd',
        vmin=0,
        vmax=2,
        aspect='auto'
    )
    ax3.set_position([0.5, 0.25, 0.2, 0.75])

    plt.subplots_adjust(left=0.06, right=0.98, bottom=0.15, top=0.95, wspace=0.12)

    save_path_png = os.path.join(save_dir, f'mode{mode_idx}_visualization_{dataset_name}.png')
    save_path_eps = os.path.join(save_dir, f'mode{mode_idx}_visualization_{dataset_name}.eps')
    plt.savefig(save_path_png, dpi=fontsize_set, bbox_inches='tight')
    plt.savefig(save_path_eps, dpi=fontsize_set, bbox_inches='tight')
    plt.close()
    print(f"Saved Mode {mode_idx} visualization to {save_path_png} and {save_path_eps}")

def mode_visualization(datasets_paths, dataset_name):

    data_path = datasets_paths[dataset_name]
    if 'Guangzhou' in dataset_name or 'Hangzhou' in dataset_name:
        dense_tensor = scipy.io.loadmat(data_path)['tensor'].transpose(0, 2, 1)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Unfold the tensor along each mode used in the paper.
    mode_data_0 = ten2mat(dense_tensor, 0) # locations x (days x time intervals)
    mode_data_1 = ten2mat(dense_tensor.transpose(1, 0, 2), 2) # days x (time intervals x locations)
    mode_data_2 = ten2mat(dense_tensor.transpose(2, 1, 0), 1) # time intervals x (days x locations)

    a_mode_0 = compute_node_correlation(mode_data_0, 6)
    a_mode_1 = compute_node_correlation(mode_data_1, 6)
    a_mode_2 = compute_node_correlation(mode_data_2, 6)

    save_mode0_pair_visualization(
        dense_tensor=dense_tensor,        mode_idx=0,
        mode_data_raw=mode_data_0,
        mode_corr_matrix=a_mode_0,
        dataset_name=dataset_name
    )

    save_mode1_pair_visualization(
        dense_tensor=dense_tensor,
        mode_idx=2,
        mode_data_raw=mode_data_1,
        mode_corr_matrix=a_mode_1,
        dataset_name=dataset_name
    )

    save_mode2_pair_visualization(
        dense_tensor=dense_tensor,
        mode_idx=1,
        mode_data_raw=mode_data_2,
        mode_corr_matrix=a_mode_2,
        dataset_name=dataset_name
    )

datasets_paths = {
    'Guangzhou': str(project_root / 'datasets' / 'Guangzhou-data-set' / 'tensor.mat'),
    }
dataset_name = 'Guangzhou'
mode_visualization(datasets_paths, dataset_name)




