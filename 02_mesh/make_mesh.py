import argparse
import os

import numpy as np
import yaml
from sfepy.discrete.fem.mesh import Mesh


def _load_geometry(spec_path):
    with open(spec_path, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    layers = {layer["name"]: float(layer["thickness_um"]) for layer in spec["layers"]}
    domain = spec["domain"]

    def _find_layer(name_fragment):
        for name, thickness in layers.items():
            if name_fragment.lower() in name.lower():
                return thickness
        raise KeyError(f"Layer with name containing '{name_fragment}' not found.")

    geom = {
        "width": float(domain["width_um"]),
        "ysz": _find_layer("ysz"),
        "tgo": _find_layer("tgo"),
        "bond": _find_layer("bond"),
        "sub": _find_layer("substrate"),
    }
    return geom


def _segment_ys(start, end, dy):
    if end <= start:
        return np.array([start])
    n = max(1, int(np.ceil((end - start) / dy)))
    return np.linspace(start, end, n + 1)


def _build_y_coords(ysz, tgo, bond, sub, dy_scale=1.0):
    # Coarser in thick layers, refined in TGO.
    # TODO: add sinusoidal interface roughness (amplitude + wavelength) here.
    dy_sub = 5.0 * dy_scale
    dy_bond = 2.0 * dy_scale
    dy_tgo = max(tgo / 3.0, 0.1) * dy_scale
    dy_ysz = 2.0 * dy_scale

    y0 = 0.0
    y1 = y0 + sub
    y2 = y1 + bond
    y3 = y2 + tgo
    y4 = y3 + ysz

    ys = []
    ys.append(_segment_ys(y0, y1, dy_sub))
    ys.append(_segment_ys(y1, y2, dy_bond)[1:])
    ys.append(_segment_ys(y2, y3, dy_tgo)[1:])
    ys.append(_segment_ys(y3, y4, dy_ysz)[1:])
    return np.concatenate(ys)


def build_mesh(spec_path, mesh_path, nx=200, dy_scale=1.0):
    geom = _load_geometry(spec_path)

    # Units: micrometers (um)
    width = geom["width"]
    ysz = geom["ysz"]
    tgo = geom["tgo"]
    bond = geom["bond"]
    sub = geom["sub"]

    # Keep x uniform, adapt y by layer.
    xs = np.linspace(0.0, width, nx + 1)
    ys = _build_y_coords(ysz, tgo, bond, sub, dy_scale=dy_scale)

    xx, yy = np.meshgrid(xs, ys)
    coors = np.c_[xx.ravel(), yy.ravel()]

    # Quad elements
    conn = []
    ny = len(ys) - 1
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n2 = n0 + (nx + 1)
            n3 = n2 + 1
            conn.append([n0, n1, n3, n2])
    conn = np.array(conn, dtype=np.int32)

    mesh = Mesh.from_data(
        "tbc_2d",
        coors,
        None,
        [conn],
        [np.full(conn.shape[0], 3, dtype=np.int32)],  # 3 = quad
        ["2_4"],
    )

    mesh.write(mesh_path)


def main():
    parser = argparse.ArgumentParser(description="Generate structured 2D mesh.")
    parser.add_argument(
        "--spec",
        default=os.path.join("00_inputs", "geometry_spec.yaml"),
        help="Path to geometry_spec.yaml",
    )
    parser.add_argument(
        "--mesh",
        default=os.path.join("02_mesh", "tbc_2d.mesh"),
        help="Output mesh path",
    )
    parser.add_argument(
        "--nx",
        type=int,
        default=200,
        help="Number of x divisions",
    )
    parser.add_argument(
        "--dy_scale",
        type=float,
        default=1.0,
        help="Scale factor for y-direction spacing",
    )
    args = parser.parse_args()

    build_mesh(args.spec, args.mesh, nx=args.nx, dy_scale=args.dy_scale)


if __name__ == "__main__":
    main()
