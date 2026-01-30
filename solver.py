from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .linalg import solve_linear_system
from .matrices import (
    formmatrixfourier, formviscoelasticmatrixfourier, formcoteskfourier, formtempcotesfourier,
    temp_pnforce_iteration
)
from .material import temp_viscoelasticmaterial
from .loading import temp_loadgeneration


def fourier_solution(GDof: int, elementdata: np.ndarray, numbernodes: int, nodecoordinate: np.ndarray,
                     matD: np.ndarray, Lz: float, rho: np.ndarray, ray: np.ndarray, Ki: np.ndarray, Gi: np.ndarray, T: np.ndarray,
                     layernumber: np.ndarray, loadfourier3D: np.ndarray, Bottom: sp.spmatrix,
                     Vertical: sp.spmatrix, interface: sp.spmatrix,
                     L: np.ndarray, n: int, t: float, Loading_index: np.ndarray,
                     temperature_data: np.ndarray | None = None):
    """Solve for all Fourier modes in L (best-effort port)."""
    temperature_data = np.asarray(temperature_data, dtype=float) if temperature_data is not None else np.zeros((1, 2), dtype=float)

    Displacement = np.zeros((GDof, n+1, len(L)), dtype=float)
    Velocity = np.zeros_like(Displacement)
    Acceleration = np.zeros_like(Displacement)

    k1, k2 = temp_viscoelasticmaterial()

    for iL, l in enumerate(L):
        l = int(l)
        # assemble matrices
        K_el, _, M, C = formmatrixfourier(GDof, elementdata, numbernodes, nodecoordinate, matD, rho, ray, Lz, l, layernumber)

        # viscoelastic additional matrices (best-effort): include cotes & tempcotes
        K_vis = sp.csr_matrix((GDof, GDof))
        if Ki.size and Gi.size:
            try:
                K_cotes = formcoteskfourier(GDof, elementdata, numbernodes, nodecoordinate, k1, k2, Ki, Gi, t, T, Lz, l, layernumber, temperature_data)
                K_temp = formtempcotesfourier(GDof, elementdata, numbernodes, nodecoordinate, t, T, Gi, Ki, k1, k2, Lz, l, layernumber, temperature_data)
                K_vis = K_cotes + K_temp
            except Exception:
                K_vis = sp.csr_matrix((GDof, GDof))

        K = K_el + K_vis + Bottom + interface
        # Newmark constants (beta=1/4, gamma=1/2)
        alpha = 4.0 / (t**2)
        alpha2 = 2.0 / t
        # effective stiffness
        K_eff = K + alpha * M + alpha2 * C + Vertical * alpha2

        # load in modal space for this l
        loading = temp_loadgeneration(GDof, n, l, loadfourier3D, Loading_index, Lz)

        # time integration
        u = np.zeros(GDof, dtype=float)
        v = np.zeros(GDof, dtype=float)
        a = np.zeros(GDof, dtype=float)
        Pnforce = np.zeros(GDof, dtype=float)

        for it in range(n+1):
            p = loading[:, it].copy()
            if it == 0:
                rhs = p + M @ (alpha*u + alpha2*v + a) + C @ (alpha2*u + v) + Vertical @ (alpha2*u + v)
            else:
                rhs = p + M @ (alpha*u + alpha2*v + a) + C @ (alpha2*u + v) + Vertical @ (alpha2*u + v) + Pnforce
            # solve
            u_new = solve_linear_system(K_eff, rhs)
            u_new = np.asarray(u_new).ravel()
            a_new = alpha * (u_new - u) - alpha2 * v - a
            v_new = v + t * (a + a_new) / 2.0

            # update Pnforce (if viscoelastic)
            if it > 0 and Ki.size and Gi.size:
                try:
                    Pnforce = temp_pnforce_iteration(numbernodes, nodecoordinate, t, T[0, :], Pnforce, K_vis, u_new, temperature_data)
                except Exception:
                    Pnforce = np.zeros(GDof, dtype=float)

            u, v, a = u_new, v_new, a_new
            Displacement[:, it, iL] = u
            Velocity[:, it, iL] = v
            Acceleration[:, it, iL] = a

    return Displacement, Velocity, Acceleration
