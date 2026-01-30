from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .temperature import temperature_shift


# ---- Q4 utilities ----
def gauss_quadrature_q4():
    locations = np.array([
        [-0.57735, -0.57735],
        [ 0.57735, -0.57735],
        [ 0.57735,  0.57735],
        [-0.57735,  0.57735],
    ], dtype=float)
    weights = np.ones(4, dtype=float)
    return weights, locations


def shape_function_q4(xi: float, eta: float):
    shape = 0.25 * np.array([
        (1 - xi) * (1 - eta),
        (1 + xi) * (1 - eta),
        (1 + xi) * (1 + eta),
        (1 - xi) * (1 + eta),
    ], dtype=float)
    natural = 0.25 * np.array([
        [-(1-eta), -(1-xi)],
        [ (1-eta), -(1+xi)],
        [ (1+eta),  (1+xi)],
        [-(1+eta),  (1-xi)],
    ], dtype=float)
    return shape, natural


def jacobian(node_xy: np.ndarray, natural_derivatives: np.ndarray):
    J = node_xy.T @ natural_derivatives
    XY = natural_derivatives @ np.linalg.inv(J)
    return J, XY


def formmatrixfourier(GDof: int, elementdata: np.ndarray, numbernodes: int, nodecoordinates: np.ndarray,
                      D_layers: np.ndarray, rho_layers: np.ndarray, ray: np.ndarray, Lz: float, l: int,
                      layernumber: np.ndarray):
    """Port of formmartixfourier.m.

    Returns:
        stiffness_elastic, stiffness_visco_like (zero here), mass, damping
    """
    weights, locations = gauss_quadrature_q4()
    gammal = l * np.pi / Lz
    K = sp.lil_matrix((GDof, GDof), dtype=float)
    M = sp.lil_matrix((GDof, GDof), dtype=float)
    C = sp.lil_matrix((GDof, GDof), dtype=float)

    # iterate elements per layer
    for layer_idx in range(len(layernumber) - 1):
        e_begin = int(layernumber[layer_idx])
        e_end = int(layernumber[layer_idx + 1])
        D = D_layers[:, :, min(layer_idx, D_layers.shape[2]-1)]
        rho = rho_layers[min(layer_idx, len(rho_layers)-1)]
        a0 = ray[0, min(layer_idx, ray.shape[1]-1)] if ray.size else 0.0
        a1 = ray[1, min(layer_idx, ray.shape[1]-1)] if ray.size else 0.0

        for e in range(e_begin, e_end):
            indice = elementdata[e, 1:5].astype(int)
            ndof = len(indice)
            elementdof = np.concatenate([indice, indice + numbernodes, indice + 2*numbernodes]).astype(int)

            for q, wt in enumerate(weights):
                xi, eta = locations[q]
                N, dN_nat = shape_function_q4(xi, eta)
                J, dN_xy = jacobian(nodecoordinates[indice, :], dN_nat)
                detJ = np.linalg.det(J)

                B = np.zeros((6, 3*ndof), dtype=float)
                B[0, 0:ndof] = dN_xy[:, 0]
                B[1, ndof:2*ndof] = dN_xy[:, 1]
                B[2, 2*ndof:3*ndof] = -N * gammal
                B[3, 0:ndof] = dN_xy[:, 1]
                B[3, ndof:2*ndof] = dN_xy[:, 0]
                B[4, ndof:2*ndof] = N * gammal
                B[4, 2*ndof:3*ndof] = dN_xy[:, 1]
                B[5, 0:ndof] = N * gammal
                B[5, 2*ndof:3*ndof] = dN_xy[:, 0]

                Nmat = np.zeros((3, 3*ndof), dtype=float)
                Nmat[0, 0:ndof] = N
                Nmat[1, ndof:2*ndof] = N
                Nmat[2, 2*ndof:3*ndof] = N

                Ke = (B.T @ D @ B) * wt * detJ * Lz/2
                Me = (Nmat.T @ Nmat) * (rho * wt * detJ * Lz/2)
                # Rayleigh damping
                Ce = a0 * Me + a1 * Ke

                # assemble
                for ii, I in enumerate(elementdof):
                    for jj, Jj in enumerate(elementdof):
                        K[I, Jj] += Ke[ii, jj]
                        M[I, Jj] += Me[ii, jj]
                        C[I, Jj] += Ce[ii, jj]

    return K.tocsr(), sp.csr_matrix((GDof, GDof)), M.tocsr(), C.tocsr()


def formviscoelasticmatrixfourier(GDof: int, elementdata: np.ndarray, numbernodes: int,
                                  nodecoordinates: np.ndarray, D: np.ndarray,
                                  Lz: float, l: int, layernumber: np.ndarray, start_layer_idx: int = 3):
    """Port of formviscoelasticmartixfourier.m (best-effort).

    start_layer_idx corresponds to MATLAB's layernumber(4)+1 range.
    """
    weights, locations = gauss_quadrature_q4()
    gammal = l * np.pi / Lz
    K = sp.lil_matrix((GDof, GDof), dtype=float)

    # MATLAB: for e=(layernumber(4)+1):layernumber(5)
    if len(layernumber) < start_layer_idx + 2:
        return K.tocsr()
    e_begin = int(layernumber[start_layer_idx])
    e_end = int(layernumber[start_layer_idx + 1])

    for e in range(e_begin, e_end):
        indice = elementdata[e, 1:5].astype(int)
        ndof = len(indice)
        elementdof = np.concatenate([indice, indice + numbernodes, indice + 2*numbernodes]).astype(int)
        for q, wt in enumerate(weights):
            xi, eta = locations[q]
            N, dN_nat = shape_function_q4(xi, eta)
            J, dN_xy = jacobian(nodecoordinates[indice, :], dN_nat)
            detJ = np.linalg.det(J)

            B = np.zeros((6, 3*ndof), dtype=float)
            B[0, 0:ndof] = dN_xy[:, 0]
            B[1, ndof:2*ndof] = dN_xy[:, 1]
            B[2, 2*ndof:3*ndof] = -N * gammal
            B[3, 0:ndof] = dN_xy[:, 1]
            B[3, ndof:2*ndof] = dN_xy[:, 0]
            B[4, ndof:2*ndof] = N * gammal
            B[4, 2*ndof:3*ndof] = dN_xy[:, 1]
            B[5, 0:ndof] = N * gammal
            B[5, 2*ndof:3*ndof] = dN_xy[:, 0]

            Ke = (B.T @ D @ B) * wt * detJ * Lz/2
            for ii, I in enumerate(elementdof):
                for jj, Jj in enumerate(elementdof):
                    K[I, Jj] += Ke[ii, jj]
    return K.tocsr()


def _cotes_calculation(K: np.ndarray, T: np.ndarray, t: float) -> float:
    temp = np.zeros_like(T, dtype=float)
    for i in range(len(T)):
        temp[i] = (1/90) * (7*np.exp(-t/T[i]) + 32*np.exp(-0.75*t/T[i]) + 12*np.exp(-0.5*t/T[i])
                            + 32*np.exp(-0.25*t/T[i]) + 7)
    return float(K[-1] + np.sum(K[:-1] * temp))


def formcoteskfourier(GDof: int, elementdata: np.ndarray, numbernodes: int, nodecoordinates: np.ndarray,
                      k1: np.ndarray, k2: np.ndarray, Kp: np.ndarray, Gp: np.ndarray,
                      t: float, TT: np.ndarray, Lz: float, l: int, layernumber: np.ndarray,
                      temperature_data: np.ndarray) -> sp.csr_matrix:
    """Port of formCotesKfourier.m (supports multiple VE layers)."""
    weights, locations = gauss_quadrature_q4()
    gammal = l * np.pi / Lz
    Kmat = sp.lil_matrix((GDof, GDof), dtype=float)

    num_ve = Kp.shape[0]
    for i in range(num_ve):
        Ki = Kp[i, :]
        Gi = Gp[i, :]
        Ti = TT[i, :]
        # map to top layers
        start = int(layernumber[-num_ve + i])
        end = int(layernumber[-num_ve + i + 1])
        for e in range(start, end):
            indice = elementdata[e, 1:5].astype(int)
            depth = float(nodecoordinates[:, 1].max() - np.mean(nodecoordinates[indice, 1]))
            Tshift = temperature_shift(temperature_data, depth)
            T_eff = Ti * Tshift
            g = 2 * _cotes_calculation(Gi, T_eff, t)
            k = 3 * _cotes_calculation(Ki, T_eff, t)
            Dloc = g * k1 + k * k2

            ndof = len(indice)
            elementdof = np.concatenate([indice, indice + numbernodes, indice + 2*numbernodes]).astype(int)

            for q, wt in enumerate(weights):
                xi, eta = locations[q]
                N, dN_nat = shape_function_q4(xi, eta)
                J, dN_xy = jacobian(nodecoordinates[indice, :], dN_nat)
                detJ = np.linalg.det(J)
                B = np.zeros((6, 3*ndof), dtype=float)
                B[0, 0:ndof] = dN_xy[:, 0]
                B[1, ndof:2*ndof] = dN_xy[:, 1]
                B[2, 2*ndof:3*ndof] = -N * gammal
                B[3, 0:ndof] = dN_xy[:, 1]
                B[3, ndof:2*ndof] = dN_xy[:, 0]
                B[4, ndof:2*ndof] = N * gammal
                B[4, 2*ndof:3*ndof] = dN_xy[:, 1]
                B[5, 0:ndof] = N * gammal
                B[5, 2*ndof:3*ndof] = dN_xy[:, 0]
                Ke = (B.T @ Dloc @ B) * wt * detJ * Lz/2
                for ii, I in enumerate(elementdof):
                    for jj, Jj in enumerate(elementdof):
                        Kmat[I, Jj] += Ke[ii, jj]
    return Kmat.tocsr()


def formtempcotesfourier(GDof: int, elementdata: np.ndarray, numbernodes: int, nodecoordinates: np.ndarray,
                         t: float, Ti: np.ndarray, Gi: np.ndarray, Ki: np.ndarray, k1: np.ndarray, k2: np.ndarray,
                         Lz: float, l: int, layernumber: np.ndarray, temperature_data: np.ndarray) -> sp.csr_matrix:
    """Port of formtempcotesfourier.m (Pn+Qn)."""
    weights, locations = gauss_quadrature_q4()
    gammal = l * np.pi / Lz
    Kmat = sp.lil_matrix((GDof, GDof), dtype=float)

    # shape normalization
    Ti = np.atleast_2d(np.asarray(Ti, dtype=float))
    Gi = np.atleast_2d(np.asarray(Gi, dtype=float))
    Ki = np.atleast_2d(np.asarray(Ki, dtype=float))
    if Gi.shape[0] != Ki.shape[0] or Gi.shape[0] != Ti.shape[0]:
        if Gi.shape[1] == Ki.shape[0] and Gi.shape[1] == Ti.shape[0]:
            Gi, Ki, Ti = Gi.T, Ki.T, Ti.T

    NL = len(layernumber) - 1
    numVE = min(Gi.shape[0], Ki.shape[0], Ti.shape[0], NL)
    if numVE == 0:
        return Kmat.tocsr()

    firstVE = max(1, NL - numVE + 1)  # 1-based layer number; we'll convert below
    # iterate ve layers (k is 1-based in MATLAB)
    for k_layer in range(firstVE, NL + 1):
        r = k_layer - firstVE  # 0-based row in Gi/Ki/Ti
        Ti_row = Ti[r, :].ravel()
        Gi_row = Gi[r, :].ravel()
        Ki_row = Ki[r, :].ravel()

        e_begin = int(layernumber[k_layer - 1])
        e_end = int(layernumber[k_layer])
        for e in range(e_begin, e_end):
            indice = elementdata[e, 1:5].astype(int)
            depth = float(nodecoordinates[:, 1].max() - np.mean(nodecoordinates[indice, 1]))
            Tshift = temperature_shift(temperature_data, depth)
            T_eff = Ti_row * Tshift

            exp1 = np.exp(-t / T_eff)
            poly = (7*exp1 + 32*np.exp(-0.75*t/T_eff) + 12*np.exp(-0.5*t/T_eff)
                    + 32*np.exp(-0.25*t/T_eff) + 7)
            cotes = (1/90) * poly
            Gf = float(np.sum(Gi_row * ((exp1 - 1) * cotes)))
            Kf = float(np.sum(Ki_row * ((exp1 - 1) * cotes)))
            Dloc = 2*Gf*k1 + 3*Kf*k2

            ndof = len(indice)
            elementdof = np.concatenate([indice, indice + numbernodes, indice + 2*numbernodes]).astype(int)
            for q, wt in enumerate(weights):
                xi, eta = locations[q]
                N, dN_nat = shape_function_q4(xi, eta)
                J, dN_xy = jacobian(nodecoordinates[indice, :], dN_nat)
                detJ = np.linalg.det(J)

                B = np.zeros((6, 3*ndof), dtype=float)
                B[0, 0:ndof] = dN_xy[:, 0]
                B[1, ndof:2*ndof] = dN_xy[:, 1]
                B[2, 2*ndof:3*ndof] = -N * gammal
                B[3, 0:ndof] = dN_xy[:, 1]
                B[3, ndof:2*ndof] = dN_xy[:, 0]
                B[4, ndof:2*ndof] = N * gammal
                B[4, 2*ndof:3*ndof] = dN_xy[:, 1]
                B[5, 0:ndof] = N * gammal
                B[5, 2*ndof:3*ndof] = dN_xy[:, 0]

                Ke = (B.T @ Dloc @ B) * wt * detJ * Lz/2
                for ii, I in enumerate(elementdof):
                    for jj, Jj in enumerate(elementdof):
                        Kmat[I, Jj] += Ke[ii, jj]

    return Kmat.tocsr()


def temp_pnforce_iteration(numbernodes: int, nodecoordinates: np.ndarray, t: float,
                           Ti: np.ndarray, Pnforce: np.ndarray, K: sp.spmatrix,
                           displacement: np.ndarray, temperature_data: np.ndarray) -> np.ndarray:
    """Port of temp_Pnforceinteration.m."""
    Pnforce = np.asarray(Pnforce, dtype=float).ravel()
    Pntemp = np.zeros_like(Pnforce)
    Ti = np.asarray(Ti, dtype=float).ravel()
    for i in range(numbernodes):
        if Pnforce[i] != 0:
            depth = float(nodecoordinates[:, 1].max() - nodecoordinates[i, 1])
            Tshift = temperature_shift(temperature_data, depth)
            T = Ti * Tshift
            # apply exponential decay term-wise
            Pntemp[i + np.array([0, 1, 2])*numbernodes] = Pnforce[i + np.array([0, 1, 2])*numbernodes] * np.exp(-t / T[0])
    Pnforce_new = Pntemp + K @ displacement
    return np.asarray(Pnforce_new).ravel()


def output_response(GDof: int, n: int, z: float, Lz: float, L: np.ndarray,
                    Displacement: np.ndarray, Velocity: np.ndarray, Acceleration: np.ndarray):
    """Sum Fourier modes at z to physical response.
    NOTE: handle l==0 as constant mode for ux/uy (otherwise sin(0)=0 kills it).
    """
    displacement = np.zeros((GDof, n + 1), dtype=float)
    velocity = np.zeros_like(displacement)
    acceleration = np.zeros_like(displacement)

    numbernodes = GDof // 3

    for i, l in enumerate(L):
        l = int(l)

        if l == 0:
            s = 1.0  # constant mode for ux/uy
            c = 1.0  # constant mode for uz
        else:
            s = np.sin(l * np.pi * z / Lz)
            c = np.cos(l * np.pi * z / Lz)

        # ux, uy use s; uz use c (same as MATLAB), but with l==0 special-cased above
        displacement[:numbernodes, :] += Displacement[:numbernodes, :, i] * s
        displacement[numbernodes:2 * numbernodes, :] += Displacement[numbernodes:2 * numbernodes, :, i] * s
        displacement[2 * numbernodes:, :] += Displacement[2 * numbernodes:, :, i] * c

        velocity[:numbernodes, :] += Velocity[:numbernodes, :, i] * s
        velocity[numbernodes:2 * numbernodes, :] += Velocity[numbernodes:2 * numbernodes, :, i] * s
        velocity[2 * numbernodes:, :] += Velocity[2 * numbernodes:, :, i] * c

        acceleration[:numbernodes, :] += Acceleration[:numbernodes, :, i] * s
        acceleration[numbernodes:2 * numbernodes, :] += Acceleration[numbernodes:2 * numbernodes, :, i] * s
        acceleration[2 * numbernodes:, :] += Acceleration[2 * numbernodes:, :, i] * c

    return displacement, velocity, acceleration



def strain3d(element_ids: np.ndarray, elementnodes: np.ndarray, numbernodes: int,
             nodecoordinates: np.ndarray, displacements: np.ndarray, L: np.ndarray,
             z: float, Lz: float) -> np.ndarray:
    """Port of strain3D.m, returning (n_elem, n_gauss(=4), 6)."""
    weights, locations = gauss_quadrature_q4()
    element_ids = np.asarray(element_ids, dtype=int).ravel()
    strain = np.zeros((len(element_ids), elementnodes.shape[1], 6), dtype=float)

    for j, e in enumerate(element_ids):
        indice = elementnodes[e, :].astype(int)
        elementdof = np.concatenate([indice, indice + numbernodes, indice + 2*numbernodes]).astype(int)
        nn = len(indice)
        for q in range(len(weights)):
            xi, eta = locations[q]
            N, dN_nat = shape_function_q4(xi, eta)
            _, dN_xy = jacobian(nodecoordinates[indice, :], dN_nat)

            for iL, l in enumerate(L):
                l = int(l)
                gammal = l * np.pi / Lz
                s = np.sin(l * np.pi * z / Lz)
                c = np.cos(l * np.pi * z / Lz)

                B = np.zeros((6, 3*nn), dtype=float)
                B[0, 0:nn] = dN_xy[:, 0] * s
                B[1, nn:2*nn] = dN_xy[:, 1] * s
                B[2, 2*nn:3*nn] = -N * gammal * s
                B[3, 0:nn] = dN_xy[:, 1] * s
                B[3, nn:2*nn] = dN_xy[:, 0] * s
                B[4, nn:2*nn] = N * gammal * c
                B[4, 2*nn:3*nn] = dN_xy[:, 1] * c
                B[5, 0:nn] = N * gammal * c
                B[5, 2*nn:3*nn] = dN_xy[:, 0] * c

                strain[j, q, :] += B @ displacements[elementdof, iL]
    return strain


def stress3d(strain: np.ndarray, D_layers: np.ndarray, layernumber: np.ndarray, n: int) -> np.ndarray:
    """Port of stress3D.m for elastic stress (best-effort)."""
    numberelements = strain.shape[0]
    ngauss = strain.shape[1]
    stress = np.zeros((numberelements, ngauss, 6, n+1), dtype=float)

    # take up to first 3 D matrices (as in MATLAB)
    for it in range(n+1):
        for layer_idx in range(min(D_layers.shape[2], len(layernumber)-1)):
            e_begin = int(layernumber[layer_idx])
            e_end = int(layernumber[layer_idx+1])
            D = D_layers[:, :, layer_idx]
            # stress = strain * D'
            stress[e_begin:e_end, :, :, it] = strain[e_begin:e_end, :, :, it] @ D.T
    return stress


def stress3d_viscoelastic(strain: np.ndarray, layernumber: np.ndarray, n: int, dt: float,
                          v: np.ndarray, Ei: np.ndarray, Ki: np.ndarray, Gi: np.ndarray,
                          T: np.ndarray, E_elastic: np.ndarray) -> np.ndarray:
    """Port of stress3D_viscoelastic.m (recursive linear viscoelastic update)."""
    numberelements = strain.shape[0]
    ngauss = strain.shape[1]
    stress_ve = np.zeros((numberelements, ngauss, 6, n+1), dtype=float)

    def build_elastic_D(Ev: float, nu: float) -> np.ndarray:
        coef = Ev / ((1+nu)*(1-2*nu))
        return coef * np.array([
            [1-nu, nu, nu, 0, 0, 0],
            [nu, 1-nu, nu, 0, 0, 0],
            [nu, nu, 1-nu, 0, 0, 0],
            [0, 0, 0, (1-2*nu)/2, 0, 0],
            [0, 0, 0, 0, (1-2*nu)/2, 0],
            [0, 0, 0, 0, 0, (1-2*nu)/2],
        ], dtype=float)

    # Layer 1 elastic
    D1 = build_elastic_D(E_elastic[0], v[0])
    startE = int(layernumber[0]); endE = int(layernumber[1])
    for it in range(n+1):
        eps = strain[startE:endE, :, :, it]
        stress_ve[startE:endE, :, :, it] = eps @ D1.T

    # Layer 2 elastic (if exists)
    if len(E_elastic) > 1 and len(layernumber) > 2:
        D2 = build_elastic_D(E_elastic[1], v[1])
        startE = int(layernumber[1]); endE = int(layernumber[2])
        for it in range(n+1):
            eps = strain[startE:endE, :, :, it]
            stress_ve[startE:endE, :, :, it] = eps @ D2.T

    # VE layers config from MATLAB (assumes layernumber has 5 entries: 0..4)
    layer_configs = [
        (2, 3, 0, 2),  # (start_layer_idx, end_layer_idx, mat_row, v_idx) 0-based
        (3, 4, 1, 3),
    ]
    for (Ls, Le, mat_row, v_idx) in layer_configs:
        if Le >= len(layernumber):
            continue
        startE = int(layernumber[Ls])
        endE = int(layernumber[Le])
        nu = float(v[v_idx])
        stress_ve = _compute_visco_stress_core(stress_ve, strain, startE, endE, n, dt, nu,
                                              Ki[mat_row, :], Gi[mat_row, :], T[mat_row, :])
    return stress_ve


def _compute_visco_stress_core(stress_ve: np.ndarray, strain: np.ndarray, startE: int, endE: int,
                               n: int, dt: float, nu: float,
                               Ki_raw: np.ndarray, Gi_raw: np.ndarray, T_raw: np.ndarray) -> np.ndarray:
    Ki_layer = np.asarray(Ki_raw, dtype=float).ravel()
    Gi_layer = np.asarray(Gi_raw, dtype=float).ravel()
    T_layer = np.asarray(T_raw, dtype=float).ravel()
    nTerms = len(T_layer)
    nElements = endE - startE

    Kinf = Ki_layer[-1]
    Ki = Ki_layer[:nTerms]
    Ginf = Gi_layer[-1]
    Gi = Gi_layer[:nTerms]

    H_vol = np.zeros((nElements, strain.shape[1], nTerms), dtype=float)
    H_dev = np.zeros((nElements, strain.shape[1], 6, nTerms), dtype=float)
    strain_prev = np.zeros((nElements, strain.shape[1], 6), dtype=float)

    beta = np.zeros(nTerms, dtype=float)
    exp_val = np.zeros(nTerms, dtype=float)
    for k in range(nTerms):
        x = dt / T_layer[k]
        exp_val[k] = np.exp(-x)
        if x < 1e-8:
            beta[k] = 1.0 - x/2
        elif x > 500:
            beta[k] = 1.0 / x
        else:
            beta[k] = (1 - exp_val[k]) / x

    for it in range(n+1):
        for idx in range(nElements):
            e = startE + idx
            for g in range(strain.shape[1]):
                eps_curr = strain[e, g, :, it].reshape(6)

                d_eps_total = eps_curr - strain_prev[idx, g, :]
                d_eps_vol = d_eps_total[0:3].sum()
                d_eps_dev = d_eps_total.copy()
                d_eps_dev[0:3] = d_eps_total[0:3] - d_eps_vol/3

                sigma_visco_vol = 0.0
                s_visco_dev = np.zeros(6, dtype=float)

                for k in range(nTerms):
                    H_vol[idx, g, k] = H_vol[idx, g, k] * exp_val[k] + Ki[k] * beta[k] * d_eps_vol
                    term_dev = np.zeros(6, dtype=float)
                    term_dev[0:3] = 2 * Gi[k] * beta[k] * d_eps_dev[0:3]
                    term_dev[3:6] = Gi[k] * beta[k] * d_eps_dev[3:6]
                    H_dev[idx, g, :, k] = H_dev[idx, g, :, k] * exp_val[k] + term_dev
                    sigma_visco_vol += H_vol[idx, g, k]
                    s_visco_dev += H_dev[idx, g, :, k]

                eps_curr_vol = eps_curr[0:3].sum()
                eps_curr_dev = eps_curr.copy()
                eps_curr_dev[0:3] = eps_curr[0:3] - eps_curr_vol/3

                sigma_total_vol = Kinf * eps_curr_vol + sigma_visco_vol
                s_total_dev = np.zeros(6, dtype=float)
                s_total_dev[0:3] = 2 * Ginf * eps_curr_dev[0:3] + s_visco_dev[0:3]
                s_total_dev[3:6] = Ginf * eps_curr_dev[3:6] + s_visco_dev[3:6]

                stress_ve[e, g, 0:3, it] = sigma_total_vol + s_total_dev[0:3]
                stress_ve[e, g, 3:6, it] = s_total_dev[3:6]

                strain_prev[idx, g, :] = eps_curr
    return stress_ve
