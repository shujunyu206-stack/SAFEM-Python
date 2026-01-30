from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def interface_matrix(interface_node: np.ndarray, nodecoordinate: np.ndarray, Lz: float,
                     Kx: np.ndarray | None = None, Kz: np.ndarray | None = None) -> sp.csr_matrix:
    """Port of Interface.m / Interface_spring().

    interface_node: (N,4) rows [b_start,b_end,t_start,t_end] inclusive node ids.
    """
    if Kx is None:
        Kx = np.array([0.41e10, 0.49e12, 0.09e12], dtype=float)
    if Kz is None:
        Kz = np.array([0.41e10, 0.49e12, 0.09e12], dtype=float)
    interface_node = np.asarray(interface_node, dtype=int)
    nodecoordinate = np.asarray(nodecoordinate, dtype=float)

    N = interface_node.shape[0]
    Ky = 15e12 * np.ones(N, dtype=float)

    nodes = nodecoordinate.shape[0]
    Gdof = 3 * nodes

    # dx along bottom surface y==0
    xs = nodecoordinate[np.isclose(nodecoordinate[:, 1], 0.0), 0]
    xs = np.sort(xs)
    dx = np.diff(xs)
    dx = np.concatenate([[0.0], dx, [0.0]])

    interface = sp.lil_matrix((Gdof, Gdof), dtype=float)

    for n in range(N):
        a1, a2, a3, a4 = interface_node[n]
        bottom = np.arange(a1, a2 + 1, dtype=int)
        top = np.arange(a3, a4 + 1, dtype=int)
        m = min(len(bottom), len(top))
        node_pair = np.column_stack([bottom[:m], top[:m]])
        for i, (nb, nt) in enumerate(node_pair):
            w = (dx[i] + dx[i + 1]) / 2
            tempx = (Lz/2) * np.array([[ Kx[min(n, len(Kx)-1)]*w, -Kx[min(n, len(Kx)-1)]*w],
                                       [-Kx[min(n, len(Kx)-1)]*w,  Kx[min(n, len(Kx)-1)]*w]], dtype=float)
            tempy = (Lz/2) * np.array([[ Ky[n]*w, -Ky[n]*w],
                                       [-Ky[n]*w,  Ky[n]*w]], dtype=float)
            tempz = (Lz/2) * np.array([[ Kz[min(n, len(Kz)-1)]*w, -Kz[min(n, len(Kz)-1)]*w],
                                       [-Kz[min(n, len(Kz)-1)]*w,  Kz[min(n, len(Kz)-1)]*w]], dtype=float)

            # x dof
            idx = [nb, nt]
            for r in range(2):
                for c in range(2):
                    interface[idx[r], idx[c]] += tempx[r, c]
                    interface[idx[r] + nodes, idx[c] + nodes] += tempy[r, c]
                    interface[idx[r] + 2*nodes, idx[c] + 2*nodes] += tempz[r, c]

    return interface.tocsr()
