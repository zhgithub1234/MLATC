"""Primal residual curves for MLATC."""

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scipy.io


script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models.mlatc import mlatc


figure_name = "figure3_convergence_analysis"
default_dataset = "Guangzhou"
default_missing_rate = 0.9
default_missing_types = ["BM", "TM2", "NodeM", "TimeM"]
epsilon_values = [1e-4]

# Fixed MLATC settings used for each missing-data case.
mlatc_configs = {
    "BM": {
        "model_type": "MLATC-D-day",
        "rho": 1e-5,
        "k_num": 3,
        "c": 1.0,
        "theta": 5,
    },
    "TM2": {
        "model_type": "MLATC-D-day",
        "rho": 1e-5,
        "k_num": 3,
        "c": 1.0,
        "theta": 5,
    },
    "NodeM": {
        "model_type": "MLATC-N-node",
        "rho": 1e-4,
        "k_num": 6,
        "c": 1.0,
        "theta": 25,
    },
    "TimeM": {
        "model_type": "MLATC-T-time",
        "rho": 1e-4,
        "k_num": 12,
        "c": 10.0,
        "theta": 20,
    },
}


def format_missing_rate(missing_rate):
    return "{:g}".format(float(missing_rate))


def default_tensor_path(dataset_name):
    return project_root / "datasets" / "{}-data-set".format(dataset_name) / "tensor.mat"


def default_missing_data_path(dataset_name, missing_type, missing_rate):
    filename = "{}.npy".format(format_missing_rate(missing_rate))
    return project_root / "datasets" / "generated_missing_data" / dataset_name / missing_type / filename


def resolve_path(path_text, base_dir):
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_dense_tensor(tensor_path):
    if not tensor_path.exists():
        raise FileNotFoundError("Original tensor file not found: {}".format(tensor_path))

    mat_data = scipy.io.loadmat(str(tensor_path))
    if "tensor" not in mat_data:
        raise KeyError("The file {} does not contain a 'tensor' variable.".format(tensor_path))
    return mat_data["tensor"].transpose(0, 2, 1)


def load_sparse_tensor(dataset_name, missing_type, missing_rate):
    missing_data_path = default_missing_data_path(dataset_name, missing_type, missing_rate)
    if not missing_data_path.exists():
        raise FileNotFoundError("Missing-data file not found: {}".format(missing_data_path))
    return np.load(str(missing_data_path))


def validate_same_shape(dense_tensor, sparse_tensor, missing_type):
    if dense_tensor.shape != sparse_tensor.shape:
        raise ValueError(
            "{} sparse tensor has shape {}, expected {}.".format(
                missing_type, sparse_tensor.shape, dense_tensor.shape
            )
        )


def run_mlatc(dense_tensor, sparse_tensor, dataset_name, missing_type, missing_rate, maxiter):
    if missing_type not in mlatc_configs:
        raise ValueError("Unsupported missing type: {}".format(missing_type))

    config = mlatc_configs[missing_type]
    rho = config["rho"]
    alpha = np.ones(3) / 3

    _, _, history = mlatc(
        dataset_name,
        dense_tensor,
        sparse_tensor,
        alpha,
        rho,
        config["c"],
        config["theta"],
        missing_type,
        missing_rate,
        config["k_num"],
        time.time(),
        config["model_type"],
        epsilon_values,
        maxiter=maxiter,
        save_outputs=True,
        return_history=True,
    )
    return history


def plot_history(history, missing_type, save_dir, metric_name, png_dpi, eps_dpi):
    if metric_name not in history:
        raise KeyError("History does not contain '{}'.".format(metric_name))

    fig = plt.figure(figsize=(10, 5))
    try:
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["font.size"] = 26
        plt.rcParams["axes.unicode_minus"] = False

        values = history[metric_name].to_numpy()
        plt.plot(np.arange(1, len(values) + 1), values, "b-", alpha=0.7, linewidth=2)
        plt.xlabel("Iterations", fontsize=30, fontweight="bold", fontname="Times New Roman")
        plt.ylabel("Primal residual", fontsize=30, fontweight="bold", fontname="Times New Roman")
        plt.grid(False)

        ax = plt.gca()
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties("Times New Roman")
            label.set_fontsize(26)
            label.set_fontweight("bold")

        fig.tight_layout()

        save_dir.mkdir(parents=True, exist_ok=True)
        eps_path = save_dir / "Pr_{}_.eps".format(missing_type)
        png_path = save_dir / "Pr_{}_.png".format(missing_type)
        fig.savefig(str(eps_path), dpi=eps_dpi, bbox_inches="tight")
        fig.savefig(str(png_path), dpi=png_dpi, bbox_inches="tight")
        return eps_path, png_path
    finally:
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=default_dataset, help="Dataset name.")
    parser.add_argument(
        "--missing-rate",
        type=float,
        default=default_missing_rate,
        help="Missing rate used to load generated missing-data npy files.",
    )
    parser.add_argument(
        "--missing-types",
        nargs="+",
        default=default_missing_types,
        choices=default_missing_types,
        help="Missing cases to run and plot.",
    )
    parser.add_argument(
        "--tensor-path",
        default=None,
        help="Optional path to tensor.mat. Relative paths are resolved from the project root.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(project_root / "results" / "figure_results" / figure_name),
        help="Output directory for generated convergence plots.",
    )
    parser.add_argument("--maxiter", type=int, default=100, help="Maximum MLATC iterations.")
    parser.add_argument(
        "--metric",
        default="Primal_residuals",
        choices=["Primal_residuals", "Tol", "RMSE"],
        help="History metric to plot.",
    )
    parser.add_argument("--png-dpi", type=int, default=300, help="PNG export resolution.")
    parser.add_argument("--eps-dpi", type=int, default=300, help="EPS export resolution.")
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(1000)

    tensor_path = (
        resolve_path(args.tensor_path, project_root)
        if args.tensor_path
        else default_tensor_path(args.dataset)
    )
    output_dir = resolve_path(args.output_dir, project_root) / args.dataset
    dense_tensor = load_dense_tensor(tensor_path)

    for missing_type in args.missing_types:
        sparse_tensor = load_sparse_tensor(args.dataset, missing_type, args.missing_rate)
        validate_same_shape(dense_tensor, sparse_tensor, missing_type)

        history = run_mlatc(
            dense_tensor,
            sparse_tensor,
            args.dataset,
            missing_type,
            args.missing_rate,
            args.maxiter,
        )
        eps_path, png_path = plot_history(
            history,
            missing_type,
            output_dir,
            args.metric,
            args.png_dpi,
            args.eps_dpi,
        )
        print("Saved {} and {}".format(eps_path, png_path))


if __name__ == "__main__":
    main()
