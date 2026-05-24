"""Imputation visualizations for slice-missing patterns."""

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from matplotlib.gridspec import GridSpec


script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
figure_name = "figure5_slice_missing"

default_dataset = "Guangzhou"
default_missing_rate = 0.3
default_start_date = "2016-08-01"

model_names = [
    "BGCP",
    "BTTF",
    "LRTC_TNN",
    "LATC",
    "LRTC_TMCP",
    "MLATC",
]

panel_labels = {
    "Original": "(1)",
    "Missing": "(2)",
    "BGCP": "(3)",
    "BTTF": "(4)",
    "LRTC_TNN": "(5)",
    "LATC": "(6)",
    "LRTC_TMCP": "(7)",
    "MLATC": "(8)",
}

missing_cases = {
    "NodeM": {
        "axis": 0,
        "view_index": 196,
        "zoom_origin": (35, 44),
        "zoom_size": (100, 4),
        "caption_name": "Location",
    },
    "TimeM": {
        "axis": 1,
        "view_index": 61,
        "zoom_origin": (135, 0),
        "zoom_size": (20, 35),
        "caption_name": "Time interval",
    },
    "DayM": {
        "axis": 2,
        "view_index": 47,
        "zoom_origin": (35, 44),
        "zoom_size": (100, 30),
        "caption_name": "Date",
    },
}

default_missing_types = ["NodeM", "TimeM", "DayM"]


def format_missing_rate(missing_rate):
    return "{:g}".format(float(missing_rate))


def default_tensor_path(dataset_name):
    return project_root / "datasets" / "{}-data-set".format(dataset_name) / "tensor.mat"


def default_sparse_path(dataset_name, missing_type, missing_rate):
    filename = "{}_{}.npz".format(missing_type, format_missing_rate(missing_rate))
    return project_root / "results" / "slice_missing_data" / dataset_name / filename


def default_recovered_dir(dataset_name, missing_type):
    return project_root / "results" / "slice_recovered_data" / dataset_name / missing_type


def generation_script_path(script_name):
    return project_root / "experiments" / script_name


def validate_default_generation_args(dataset_name, missing_rate):
    if dataset_name != default_dataset or format_missing_rate(missing_rate) != format_missing_rate(default_missing_rate):
        raise FileNotFoundError(
            "Figure 5 data generation scripts only support {} with missing rate {}.".format(
                default_dataset,
                format_missing_rate(default_missing_rate),
            )
        )


def build_generation_message(missing_paths, script_name):
    script_path = generation_script_path(script_name)
    return (
        "Missing required file(s):\n{}\n\n"
        "Please run this script first:\npython \"{}\""
    ).format("\n".join(str(path) for path in missing_paths), script_path)


def build_recovered_generation_message(missing_paths):
    message = build_generation_message(
        missing_paths,
        "generate_models_recovered_data_for_slicemissing.py",
    )
    if any(path.name.startswith("LRTC_TMCP_") for path in missing_paths):
        message += (
            "\n\nNote: LRTC_TMCP is restored in the Figure 5 panel list, "
            "but no LRTC_TMCP implementation exists in this project. "
            "Please provide its recovered npz file under results/slice_recovered_data."
        )
    return message


def missing_sparse_paths(dataset_name, missing_types, missing_rate):
    return [
        default_sparse_path(dataset_name, missing_type, missing_rate)
        for missing_type in missing_types
        if not default_sparse_path(dataset_name, missing_type, missing_rate).exists()
    ]


def missing_recovered_paths(dataset_name, missing_types, missing_rate):
    missing_paths = []
    for missing_type in missing_types:
        recovered_dir = default_recovered_dir(dataset_name, missing_type)
        for model_name in model_names:
            filename = "{}_{}_{}.npz".format(
                model_name, missing_type, format_missing_rate(missing_rate)
            )
            npz_path = recovered_dir / filename
            if not npz_path.exists():
                missing_paths.append(npz_path)
    return missing_paths


def ensure_slice_missing_data(dataset_name, missing_types, missing_rate):
    missing_paths = missing_sparse_paths(dataset_name, missing_types, missing_rate)
    if not missing_paths:
        return

    validate_default_generation_args(dataset_name, missing_rate)
    raise FileNotFoundError(
        build_generation_message(missing_paths, "generate_slice_missing_data.py")
    )


def ensure_recovered_data(dataset_name, missing_types, missing_rate):
    missing_paths = missing_recovered_paths(dataset_name, missing_types, missing_rate)
    if not missing_paths:
        return

    validate_default_generation_args(dataset_name, missing_rate)
    raise FileNotFoundError(
        build_recovered_generation_message(missing_paths)
    )


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

    dense_tensor = mat_data["tensor"].transpose(0, 2, 1)
    if dense_tensor.ndim != 3:
        raise ValueError("Expected a 3-D tensor, got shape {}.".format(dense_tensor.shape))
    return dense_tensor


def load_npz_array(npz_path, key):
    if not npz_path.exists():
        raise FileNotFoundError("Required npz file not found: {}".format(npz_path))

    with np.load(str(npz_path)) as data:
        if key not in data.files:
            raise KeyError("The file {} does not contain '{}'.".format(npz_path, key))
        return data[key]


def validate_tensor_shape(tensor, expected_shape, tensor_name):
    if tensor.shape != expected_shape:
        raise ValueError(
            "{} has shape {}, expected {}.".format(tensor_name, tensor.shape, expected_shape)
        )


def load_recovered_tensors(dataset_name, missing_type, missing_rate, expected_shape):
    recovered_dir = default_recovered_dir(dataset_name, missing_type)
    tensors = {}
    missing_files = []

    for model_name in model_names:
        filename = "{}_{}_{}.npz".format(
            model_name, missing_type, format_missing_rate(missing_rate)
        )
        npz_path = recovered_dir / filename
        if not npz_path.exists():
            missing_files.append(str(npz_path))
            continue

        tensor_hat = load_npz_array(npz_path, "tensor_hat")
        validate_tensor_shape(tensor_hat, expected_shape, model_name)
        tensors[model_name] = tensor_hat

    if missing_files:
        raise FileNotFoundError(
            "Missing recovered tensor file(s):\n{}".format("\n".join(missing_files))
        )

    return tensors


def get_case_config(missing_type):
    if missing_type not in missing_cases:
        raise ValueError(
            "Unsupported missing type '{}'. Choose from {}.".format(
                missing_type, ", ".join(default_missing_types)
            )
        )
    return missing_cases[missing_type]


def validate_index(index, size, description):
    if index < 0 or index >= size:
        raise ValueError("{} index {} is outside valid range [0, {}).".format(description, index, size))


def validate_zoom_window(matrix_shape, row0, col0, height, width):
    row1 = row0 + height
    col1 = col0 + width
    if row0 < 0 or col0 < 0 or row1 > matrix_shape[0] or col1 > matrix_shape[1]:
        raise ValueError(
            "Zoom window rows [{}, {}) and columns [{}, {}) does not fit matrix shape {}.".format(
                row0, row1, col0, col1, matrix_shape
            )
        )


def select_matrix(tensor, axis, index):
    if axis == 0:
        return tensor[index, :, :]
    if axis == 1:
        return tensor[:, index, :]
    if axis == 2:
        return tensor[:, :, index]
    raise ValueError("Unsupported tensor axis: {}".format(axis))


def generate_matrix_data(dense_tensor, recovered_tensor, missing_type):
    config = get_case_config(missing_type)
    axis = config["axis"]
    index = config["view_index"]
    validate_index(index, dense_tensor.shape[axis], config["caption_name"])

    mat_recovered = select_matrix(recovered_tensor, axis, index)
    mat_true = select_matrix(dense_tensor, axis, index)

    row0, col0 = config["zoom_origin"]
    height, width = config["zoom_size"]
    validate_zoom_window(mat_recovered.shape, row0, col0, height, width)

    mat_zoom = mat_recovered[row0 : row0 + height, col0 : col0 + width]
    mat_true_zoom = mat_true[row0 : row0 + height, col0 : col0 + width]
    mat_residual = np.abs(mat_true_zoom - mat_zoom)

    return {
        "full": mat_recovered,
        "zoom": mat_zoom,
        "residual": mat_residual,
        "row0": row0,
        "col0": col0,
        "index": index,
    }


def build_caption(missing_type, index, start_date_text):
    config = get_case_config(missing_type)
    if missing_type == "DayM":
        start_date = datetime.strptime(start_date_text, "%Y-%m-%d").date()
        return (start_date + timedelta(days=index)).strftime("%Y-%m-%d")
    return "{} {}".format(config["caption_name"], index)


def plot_three_subplots(
    matrix_data,
    dataset_name,
    mode_name,
    missing_type,
    output_dir,
    start_date,
    cmap_name,
    vmin,
    vmax,
    png_dpi,
    eps_dpi,
):
    fig = plt.figure(figsize=(10, 8))

    try:
        gs = GridSpec(2, 2, height_ratios=[3, 5])

        ax_zoom = fig.add_subplot(gs[0, 0])
        ax_zoom.imshow(matrix_data["zoom"], cmap=cmap_name, aspect="auto", vmin=vmin, vmax=vmax)
        ax_zoom.axis("off")

        ax_res = fig.add_subplot(gs[0, 1])
        ax_res.imshow(matrix_data["residual"], cmap=cmap_name, aspect="auto", vmin=vmin, vmax=vmax)
        ax_res.axis("off")

        ax_full = fig.add_subplot(gs[1, :])
        ax_full.imshow(matrix_data["full"], cmap=cmap_name, aspect="auto", vmin=vmin, vmax=vmax)
        ax_full.axis("off")

        detail_caption = build_caption(missing_type, matrix_data["index"], start_date)
        caption_text = panel_labels.get(mode_name, detail_caption)
        ax_full.text(
            0.5,
            -0.15,
            caption_text,
            ha="center",
            va="center",
            transform=ax_full.transAxes,
            fontsize=55,
            fontweight="bold",
        )

        rect = plt.Rectangle(
            (matrix_data["col0"], matrix_data["row0"]),
            width=matrix_data["zoom"].shape[1],
            height=matrix_data["zoom"].shape[0],
            edgecolor="red",
            facecolor="none",
            linewidth=6,
        )
        ax_full.add_patch(rect)

        fig.tight_layout()

        save_dir = output_dir / dataset_name / missing_type
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_dir / "{}.png".format(mode_name)), dpi=png_dpi, bbox_inches="tight")
        fig.savefig(str(save_dir / "{}.eps".format(mode_name)), dpi=eps_dpi, bbox_inches="tight")
    finally:
        plt.close(fig)


def save_colorbar(output_dir, dataset_name, cmap_name, vmin, vmax, png_dpi, eps_dpi):
    save_dir = output_dir / dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10, 8))
    try:
        gs = fig.add_gridspec(1, 3, width_ratios=[0.47, 0.06, 0.47])

        ax_left = fig.add_subplot(gs[0, 0])
        ax_left.axis("off")

        ax_colorbar = fig.add_subplot(gs[0, 1])
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        scalar_map = plt.cm.ScalarMappable(cmap=cmap_name, norm=norm)
        scalar_map.set_array([])
        cbar = plt.colorbar(
            scalar_map,
            cax=ax_colorbar,
            orientation="vertical",
            ticks=np.linspace(vmin, vmax, 5),
        )
        cbar.ax.tick_params(labelsize=45)

        ax_right = fig.add_subplot(gs[0, 2])
        ax_right.axis("off")

        for label in cbar.ax.yaxis.get_ticklabels():
            label.set_fontname("Times New Roman")

        fig.savefig(str(save_dir / "colorbar.eps"), dpi=eps_dpi, bbox_inches="tight", pad_inches=0)
        fig.savefig(str(save_dir / "colorbar.png"), dpi=png_dpi, bbox_inches="tight", pad_inches=0)
    finally:
        plt.close(fig)


def plot_slice_result(
    dense_tensor,
    recovered_tensor,
    dataset_name,
    mode_name,
    missing_type,
    output_dir,
    start_date,
    cmap_name,
    vmin,
    vmax,
    png_dpi,
    eps_dpi,
):
    matrix_data = generate_matrix_data(dense_tensor, recovered_tensor, missing_type)
    plot_three_subplots(
        matrix_data,
        dataset_name,
        mode_name,
        missing_type,
        output_dir,
        start_date,
        cmap_name,
        vmin,
        vmax,
        png_dpi,
        eps_dpi,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=default_dataset, help="Dataset name.")
    parser.add_argument(
        "--missing-rate",
        type=float,
        default=default_missing_rate,
        help="Missing rate used in the input npz filenames.",
    )
    parser.add_argument(
        "--missing-types",
        nargs="+",
        default=default_missing_types,
        choices=default_missing_types,
        help="Slice-missing cases to plot.",
    )
    parser.add_argument(
        "--tensor-path",
        default=None,
        help="Optional path to tensor.mat. Relative paths are resolved from the project root.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(project_root / "results" / "figure_results" / figure_name),
        help="Output directory for generated plots and reports.",
    )
    parser.add_argument(
        "--start-date",
        default=default_start_date,
        help="Start date for day-slice captions, formatted as YYYY-MM-DD.",
    )
    parser.add_argument("--cmap", default="jet", help="Matplotlib colormap name.")
    parser.add_argument("--vmin", type=float, default=0.0, help="Minimum heatmap value.")
    parser.add_argument("--vmax", type=float, default=60.0, help="Maximum heatmap value.")
    parser.add_argument("--png-dpi", type=int, default=300, help="PNG export resolution.")
    parser.add_argument("--eps-dpi", type=int, default=300, help="EPS export resolution.")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_name = args.dataset
    tensor_path = (
        resolve_path(args.tensor_path, project_root)
        if args.tensor_path
        else default_tensor_path(dataset_name)
    )
    output_dir = resolve_path(args.output_dir, project_root)

    dense_tensor = load_dense_tensor(tensor_path)
    ensure_slice_missing_data(dataset_name, args.missing_types, args.missing_rate)
    ensure_recovered_data(dataset_name, args.missing_types, args.missing_rate)

    for missing_type in args.missing_types:
        sparse_path = default_sparse_path(dataset_name, missing_type, args.missing_rate)
        sparse_tensor = load_npz_array(sparse_path, "sparse_tensor")
        validate_tensor_shape(sparse_tensor, dense_tensor.shape, "sparse_tensor")

        recovered_tensors = load_recovered_tensors(
            dataset_name, missing_type, args.missing_rate, dense_tensor.shape
        )

        plot_slice_result(
            dense_tensor,
            dense_tensor,
            dataset_name,
            "Original",
            missing_type,
            output_dir,
            args.start_date,
            args.cmap,
            args.vmin,
            args.vmax,
            args.png_dpi,
            args.eps_dpi,
        )
        plot_slice_result(
            dense_tensor,
            sparse_tensor,
            dataset_name,
            "Missing",
            missing_type,
            output_dir,
            args.start_date,
            args.cmap,
            args.vmin,
            args.vmax,
            args.png_dpi,
            args.eps_dpi,
        )

        for model_name in model_names:
            plot_slice_result(
                dense_tensor,
                recovered_tensors[model_name],
                dataset_name,
                model_name,
                missing_type,
                output_dir,
                args.start_date,
                args.cmap,
                args.vmin,
                args.vmax,
                args.png_dpi,
                args.eps_dpi,
            )

    save_colorbar(
        output_dir,
        dataset_name,
        args.cmap,
        args.vmin,
        args.vmax,
        args.png_dpi,
        args.eps_dpi,
    )
    print("Saved {} outputs to {}".format(figure_name, output_dir / dataset_name))


if __name__ == "__main__":
    main()
