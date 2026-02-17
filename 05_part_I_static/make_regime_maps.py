import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _heatmap(df, x_col, y_col, z_col, bins, out_path, title):
    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()
    z = df[z_col].to_numpy()

    x_edges = np.linspace(np.nanmin(x), np.nanmax(x), bins + 1)
    y_edges = np.linspace(np.nanmin(y), np.nanmax(y), bins + 1)
    z_grid = np.full((bins, bins), np.nan)

    for i in range(bins):
        for j in range(bins):
            mask = (
                (x >= x_edges[i])
                & (x < x_edges[i + 1])
                & (y >= y_edges[j])
                & (y < y_edges[j + 1])
            )
            if np.any(mask):
                z_grid[j, i] = np.nanmean(z[mask])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        z_grid,
        origin="lower",
        aspect="auto",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
    )
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate Part I regime maps.")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset CSV with interface metrics",
    )
    parser.add_argument("--bins", type=int, default=20)
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    out_dir = Path("06_regime_maps")
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = [
        "ysz_tgo_max_sigma_yy",
        "ysz_tgo_mean_sed",
    ]
    pairs = [
        ("delta_t", "e_scale"),
        ("delta_t", "alpha_scale"),
        ("e_scale", "alpha_scale"),
    ]

    for z_col in outputs:
        if z_col not in df.columns:
            continue
        for x_col, y_col in pairs:
            if x_col not in df.columns or y_col not in df.columns:
                continue
            out_path = out_dir / f"heatmap_{z_col}_{x_col}_vs_{y_col}.png"
            title = f"{z_col}: {x_col} vs {y_col}"
            _heatmap(df, x_col, y_col, z_col, args.bins, out_path, title)


if __name__ == "__main__":
    main()
