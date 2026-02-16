import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root / "02_mesh"))
sys.path.append(str(repo_root / "03_solver"))

from make_mesh import build_mesh  # noqa: E402
from thermoelastic_solver import build_case_context, solve_delta_t  # noqa: E402
from extract_features import extract_features  # noqa: E402


RESULT_ROOT = Path("05_results") / "02_part_I_static_ranking"
OAT_DIR = RESULT_ROOT / "01_oat_sweeps"
LHS_DIR = RESULT_ROOT / "02_lhs_dataset"
ML_DIR = RESULT_ROOT / "03_ml_models"
IMPORTANCE_DIR = RESULT_ROOT / "04_feature_importance"
INTERACTIONS_DIR = RESULT_ROOT / "05_interactions"
SUMMARY_DIR = RESULT_ROOT / "summary_tables"

INTERFACES = ["ysz_tgo", "tgo_bc"]
METRICS = ["max_sigma_yy", "max_tau_xy", "mean_sed"]


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def ensure_output_tree():
    for folder in [
        OAT_DIR,
        LHS_DIR,
        ML_DIR,
        IMPORTANCE_DIR,
        INTERACTIONS_DIR,
        SUMMARY_DIR,
    ]:
        ensure_dir(folder)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_csv(rows, path):
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def metric_key(interface, metric):
    return f"{interface}_{metric}"


def extract_case_metrics(features):
    out = {}
    for interface in INTERFACES:
        for metric in METRICS:
            key = metric_key(interface, metric)
            out[key] = features.get(key)
    return out


def plot_triplet(x, series, xlabel, title, out_path, invert_x=False):
    ensure_dir(Path(out_path).parent)
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(6, 9))
    labels = {
        "max_sigma_yy": "sigma_yy (Pa)",
        "max_tau_xy": "|tau_xy| (Pa)",
        "mean_sed": "mean SED (J/m^3)",
    }

    for metric_idx, metric in enumerate(METRICS):
        for y, label in series[metric]:
            axes[metric_idx].plot(x, y, marker="o", label=label)
        axes[metric_idx].set_ylabel(labels[metric])
        if series[metric]:
            axes[metric_idx].legend()

    axes[-1].set_xlabel(xlabel)
    fig.suptitle(title)
    if invert_x:
        for ax in axes:
            ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def lhs_samples(bounds, n_samples, rng):
    low = float(bounds[0])
    high = float(bounds[1])
    if n_samples <= 1:
        return [low]
    edges = np.linspace(0.0, 1.0, n_samples + 1)
    u = rng.uniform(edges[:-1], edges[1:])
    rng.shuffle(u)
    return (low + (high - low) * u).tolist()


def _find_layer_thickness(layers, name_fragment):
    for layer in layers:
        if name_fragment.lower() in layer["name"].lower():
            return float(layer["thickness_um"])
    raise KeyError(f"Layer with name containing '{name_fragment}' not found.")


def _set_layer_thickness(layers, name_fragment, thickness):
    for layer in layers:
        if name_fragment.lower() in layer["name"].lower():
            layer["thickness_um"] = float(thickness)
            return
    raise KeyError(f"Layer with name containing '{name_fragment}' not found.")


def _scale_materials(base_mats, alpha_scale, e_scale):
    mats = yaml.safe_load(yaml.safe_dump(base_mats))
    for mat in mats["materials"].values():
        if "alpha_1K" in mat:
            mat["alpha_1K"] = float(mat["alpha_1K"]) * float(alpha_scale)
        if "E_GPa" in mat:
            mat["E_GPa"] = float(mat["E_GPa"]) * float(e_scale)
    return mats


def run_case(
    geometry_path,
    materials_path,
    mesh_path,
    delta_t,
    tgo_thickness_um=None,
    ysz_thickness_um=None,
    bond_thickness_um=None,
    alpha_scale=1.0,
    e_scale=1.0,
    enable_roughness=False,
    roughness_amplitude=0.0,
    roughness_wavelength=100.0,
    nx=200,
    dy_scale=1.0,
    n_select=200,
):
    base_geom = load_yaml(geometry_path)
    base_mats = load_yaml(materials_path)

    geom = dict(base_geom)
    geom_layers = [dict(layer) for layer in base_geom["layers"]]
    geom["layers"] = geom_layers
    if tgo_thickness_um is not None:
        _set_layer_thickness(geom["layers"], "tgo", tgo_thickness_um)
    if ysz_thickness_um is not None:
        _set_layer_thickness(geom["layers"], "ysz", ysz_thickness_um)
    if bond_thickness_um is not None:
        _set_layer_thickness(geom["layers"], "bond", bond_thickness_um)

    mats = _scale_materials(base_mats, alpha_scale, e_scale)

    ensure_dir(Path(mesh_path).parent)
    geom_path = Path(mesh_path).with_suffix(".geometry.yaml")
    mats_path = Path(mesh_path).with_suffix(".materials.yaml")
    with open(geom_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(geom, f, sort_keys=False)
    with open(mats_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(mats, f, sort_keys=False)

    build_mesh(
        str(geom_path),
        str(mesh_path),
        nx=nx,
        dy_scale=dy_scale,
        enable_roughness=enable_roughness,
        roughness_amplitude=roughness_amplitude,
        roughness_wavelength=roughness_wavelength,
    )

    context = build_case_context(str(geom_path), str(mats_path), str(mesh_path))
    pb, state, out = solve_delta_t(context, delta_t, growth_strain=0.0, bc_variant="fixed")
    features = extract_features(
        pb,
        regions=context["regions"],
        materials={
            "substrate": {
                "lam": context["props"]["substrate"]["lam"],
                "mu": context["props"]["substrate"]["mu"],
            },
            "bondcoat": {
                "lam": context["props"]["bondcoat"]["lam"],
                "mu": context["props"]["bondcoat"]["mu"],
            },
            "tgo": {
                "lam": context["props"]["tgo"]["lam"],
                "mu": context["props"]["tgo"]["mu"],
            },
            "ysz": {
                "lam": context["props"]["ysz"]["lam"],
                "mu": context["props"]["ysz"]["mu"],
            },
        },
        y2=context["y2"],
        y3=context["y3"],
        output_csv=os.devnull,
        delta_t=delta_t,
        n_select=n_select,
    )
    return features


def save_input_histograms(df, out_dir):
    ensure_dir(out_dir)
    for col in df.columns:
        plt.figure()
        plt.hist(df[col], bins=20, edgecolor="k")
        plt.xlabel(col)
        plt.ylabel("count")
        plt.title(f"Input distribution: {col}")
        plt.tight_layout()
        plt.savefig(Path(out_dir) / f"hist_{col}.png", dpi=200)
        plt.close()


def save_input_correlation(df, out_path):
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, cmap="viridis", vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(corr.columns)))
    ax.set_yticklabels(corr.columns)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
