from pathlib import Path

import numpy as np
import scipy.io


project_root = Path(__file__).resolve().parents[1]
datasets_root = project_root / "datasets"
results_root = project_root / "results"
dataset_name = "Guangzhou"
missing_rate = 0.3
missing_types = ["NodeM", "TimeM", "DayM"]
random_seed = 1000


def load_dense_tensor(dataset):
    dataset_paths = {
        "Guangzhou": datasets_root / "Guangzhou-data-set" / "tensor.mat",
        "Hangzhou": datasets_root / "Hangzhou-data-set" / "tensor.mat",
        "Seattle": datasets_root / "Seattle-data-set" / "tensor.npz",
    }
    data_path = dataset_paths[dataset]

    if dataset in {"Guangzhou", "Hangzhou"}:
        return scipy.io.loadmat(str(data_path))["tensor"].transpose(0, 2, 1)
    if dataset == "Seattle":
        return np.load(str(data_path))["arr_0"].transpose(0, 2, 1)

    raise ValueError(f"Unknown dataset: {dataset}")


def generate_slice_missing_tensor(dense_tensor, missing_type, rate):
    dim1, dim2, dim3 = dense_tensor.shape
    np.random.seed(random_seed)

    if missing_type == "NodeM":
        mask = np.round(np.random.rand(dim1) + 0.5 - rate)[:, None, None]
    elif missing_type == "TimeM":
        mask = np.round(np.random.rand(dim2) + 0.5 - rate)[None, :, None]
    elif missing_type == "DayM":
        mask = np.round(np.random.rand(dim3) + 0.5 - rate)[None, None, :]
    else:
        raise ValueError(f"Unknown slice missing type: {missing_type}")

    return dense_tensor * mask


def save_slice_missing_data(dataset, rate, selected_missing_types):
    dense_tensor = load_dense_tensor(dataset)
    output_dir = results_root / "slice_missing_data" / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    for missing_type in selected_missing_types:
        sparse_tensor = generate_slice_missing_tensor(dense_tensor, missing_type, rate)
        output_path = output_dir / f"{missing_type}_{rate:g}.npz"
        np.savez(str(output_path), sparse_tensor=sparse_tensor)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    save_slice_missing_data(dataset_name, missing_rate, missing_types)
