from __future__ import annotations

import numpy as np


def temp_viscoelasticmaterial() -> tuple[np.ndarray, np.ndarray]:
    k1 = np.array([
        [2/3, -1/3, -1/3, 0, 0, 0],
        [-1/3, 2/3, -1/3, 0, 0, 0],
        [-1/3, -1/3, 2/3, 0, 0, 0],
        [0, 0, 0, 1/2, 0, 0],
        [0, 0, 0, 0, 1/2, 0],
        [0, 0, 0, 0, 0, 1/2],
    ], dtype=float)
    k2 = np.array([
        [1/3, 1/3, 1/3, 0, 0, 0],
        [1/3, 1/3, 1/3, 0, 0, 0],
        [1/3, 1/3, 1/3, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ], dtype=float)
    return k1, k2


def material(Einf1: float, Ei_1_in: np.ndarray, Einf2: float, Ei_2_in: np.ndarray, Ee_in: np.ndarray, dampingfactor_in: np.ndarray):
    """Port of Material.m.

    Returns:
        rho, v, E, D, ray, Ei, Ki, Gi, T
    """
    rho = np.array([1800, 2400, 2400, 2400], dtype=float)
    v = np.array([0.4, 0.35, 0.35, 0.35], dtype=float)
    dampingfactor_in = np.asarray(dampingfactor_in, dtype=float).ravel()
    ray = np.vstack([dampingfactor_in, dampingfactor_in])

    E = np.asarray(Ee_in, dtype=float).ravel()
    D = np.zeros((6, 6, len(v)), dtype=float)

    for i in range(len(E)):
        vi = v[i]
        Ei_val = E[i]
        coef = Ei_val / ((1 + vi) * (1 - 2 * vi))
        D[:, :, i] = coef * np.array([
            [1-vi, vi, vi, 0, 0, 0],
            [vi, 1-vi, vi, 0, 0, 0],
            [vi, vi, 1-vi, 0, 0, 0],
            [0, 0, 0, (1-2*vi)/2, 0, 0],
            [0, 0, 0, 0, (1-2*vi)/2, 0],
            [0, 0, 0, 0, 0, (1-2*vi)/2],
        ], dtype=float)

    Ei_1_in = np.asarray(Ei_1_in, dtype=float).ravel()
    Ei_2_in = np.asarray(Ei_2_in, dtype=float).ravel()

    Ei_ve_all = np.vstack([
        np.concatenate([Ei_2_in, [Einf2]]),
        np.concatenate([Ei_1_in, [Einf1]]),
    ])

    Ki = np.zeros_like(Ei_ve_all)
    Gi = np.zeros_like(Ei_ve_all)

    T = np.array([
        [0.000000019, 0.00000002, 0.000002, 0.0002, 0.02, 2, 200, 20000],
        [0.000000018, 0.000000021, 0.0000021, 0.00021, 0.0201, 2, 201, 20001],
    ], dtype=float)

    for i in range(len(E), len(v)):
        row_idx = min(i - len(E), Ei_ve_all.shape[0]-1)
        vi = v[i]
        Ki[row_idx, :] = Ei_ve_all[row_idx, :] / (3 * (1 - 2 * vi))
        Gi[row_idx, :] = Ei_ve_all[row_idx, :] / (2 * (1 + vi))
        E0 = Ei_ve_all[row_idx, :].sum()
        coef = E0 / ((1 + vi) * (1 - 2 * vi))
        D[:, :, i] = coef * np.array([
            [1-vi, vi, vi, 0, 0, 0],
            [vi, 1-vi, vi, 0, 0, 0],
            [vi, vi, 1-vi, 0, 0, 0],
            [0, 0, 0, (1-2*vi)/2, 0, 0],
            [0, 0, 0, 0, (1-2*vi)/2, 0],
            [0, 0, 0, 0, 0, (1-2*vi)/2],
        ], dtype=float)

    Ei_out = Ei_ve_all
    return rho, v, E, D, ray, Ei_out, Ki, Gi, T
