import numpy as np
from sfepy.discrete.fem.mesh import Mesh


def main():
    # Units: micrometers (um)
    width = 2000.0
    ysz = 200.0
    tgo = 1.0
    bond = 150.0
    sub = 1000.0

    height = ysz + tgo + bond + sub

    # Coarse for now. We refine later.
    nx, ny = 200, 150

    xs = np.linspace(0.0, width, nx + 1)
    ys = np.linspace(0.0, height, ny + 1)

    xx, yy = np.meshgrid(xs, ys)
    coors = np.c_[xx.ravel(), yy.ravel()]

    # Quad elements
    conn = []
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

    mesh.write("02_mesh/tbc_2d.mesh")


if __name__ == "__main__":
    main()
