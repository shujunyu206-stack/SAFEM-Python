from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def boundary(E: np.ndarray, Ei: np.ndarray, rho: np.ndarray, v: np.ndarray,
             GDof: int, numbernode: int, nodecoordinate: np.ndarray,
             Dx: np.ndarray, Dy: np.ndarray, Lz: float) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Port of Boundary.m -> Bottom_stiffness, Vertical (best-effort)."""
    bottom = bottomcondition_stiffness(E[0], v[0], GDof, numbernode, Dx, nodecoordinate, Lz)
    vertical = verticalcondition(E, v, rho, GDof, nodecoordinate, Lz)
    return bottom.tocsr(), vertical.tocsr()


def bottomcondition_stiffness(E: float, v: float, GDof: int, numbernodes: int,
                             Dx: np.ndarray, nodecoordinates: np.ndarray, Lz: float) -> sp.lil_matrix:
    """Port of bottomcondition_stiffness() inside Boundary.m."""
    Dx = np.asarray(Dx, dtype=float).ravel()
    Kv = 2.35 * E * np.sqrt(Dx / Lz) / (1 - v**2) / 2
    Kh = 4.6 * E * np.sqrt(Dx / Lz) / ((1 + v) * (2 - v)) / 2

    bottom = sp.lil_matrix((GDof, GDof), dtype=float)
    indice = np.where(np.isclose(nodecoordinates[:, 1], 0.0))[0]

    Kv_left = np.concatenate([[0.0], Kv])
    Kv_right = np.concatenate([Kv, [0.0]])
    temp_Kv = (Kv_left + Kv_right) * Lz / 2

    Kh_left = np.concatenate([[0.0], Kh])
    Kh_right = np.concatenate([Kh, [0.0]])
    temp_Kh = (Kh_left + Kh_right) * Lz / 2

    # MATLAB:
    # bottom(indice+numbernodes,indice+numbernodes)=Kv;  (y dof)
    # bottom(indice,indice)=Kh; (x dof)
    # bottom(indice+2*numbernodes,indice+2*numbernodes)=Kh; (z dof)
    for i, nid in enumerate(indice):
        bottom[nid + numbernodes, nid + numbernodes] += temp_Kv[i]
        bottom[nid, nid] += temp_Kh[i]
        bottom[nid + 2 * numbernodes, nid + 2 * numbernodes] += temp_Kh[i]
    return bottom


def verticalcondition(E: np.ndarray, v: np.ndarray, rho: np.ndarray, GDof: int,
                      nodecoordinates: np.ndarray, Lz: float) -> sp.lil_matrix:
    """Port of verticalcondition() inside Boundary.m.

    MATLAB uses per-layer dy segments. Here we infer dy from boundary-node spacing and
    assign layer properties by depth using piecewise partitioning of the total height.
    """
    numbernodes = nodecoordinates.shape[0]
    nodes = GDof // 3
    vertical = sp.lil_matrix((GDof, GDof), dtype=float)

    x0 = nodecoordinates[:, 0].min()
    x1 = nodecoordinates[:, 0].max()
    # boundary nodes (two vertical edges)
    indice1 = np.where(np.isclose(nodecoordinates[:, 0], x0))[0]
    indice2 = np.where(np.isclose(nodecoordinates[:, 0], x1))[0]
    # sort by y
    indice1 = indice1[np.argsort(nodecoordinates[indice1, 1])]
    indice2 = indice2[np.argsort(nodecoordinates[indice2, 1])]
    yb = nodecoordinates[indice1, 1]
    dy = np.diff(yb)
    # pad dy left/right as MATLAB ([0 Dy]+[Dy 0])/2
    dy_pad = (np.concatenate([[0.0], dy]) + np.concatenate([dy, [0.0]])) / 2

    H = float(nodecoordinates[:, 1].max() - nodecoordinates[:, 1].min())
    nlayer = len(E)
    # layer boundaries equally partitioned (fallback)
    bounds = np.linspace(0.0, H, nlayer + 1)

    for k in range(len(indice1)):
        y = yb[k]
        # find layer index
        li = int(np.clip(np.searchsorted(bounds, y, side='right') - 1, 0, nlayer - 1))
        Vp = np.sqrt(E[li] * (1 - v[li]) / ((1 + v[li]) * (1 - 2 * v[li]) * rho[li]))
        Vs = np.sqrt(E[li] / (2 * (1 + v[li]) * rho[li]))
        coef = Lz * dy_pad[k] * rho[li] / 2
        # x-dof uses Vp, y/z use Vs in MATLAB
        for nid in (indice1[k], indice2[k]):
            vertical[nid, nid] += coef * Vp
            vertical[nid + nodes, nid + nodes] += coef * Vs
            vertical[nid + 2 * nodes, nid + 2 * nodes] += coef * Vs

    return vertical
