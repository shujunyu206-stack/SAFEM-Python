from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class LoadingResult:
    loadfourier3D: np.ndarray  # (nL, 1+6*(n+1), ribs)
    Lz: float
    n: int
    t: float
    V: float


def temp_interp_loading(load_raw: np.ndarray, Lz: float, type_: int = 1) -> np.ndarray:
    """Port of temp_interp_loading.m."""
    load_raw = np.asarray(load_raw, dtype=float)
    load_raw = load_raw[np.argsort(load_raw[:, 0])]
    _, idx = np.unique(load_raw[:, 0], return_index=True)
    idx = np.sort(idx)
    xp = load_raw[idx, 0]
    fp = load_raw[idx, 1]

    xq = np.arange(0.0, Lz + 1e-12, 0.01)
    vq = np.interp(xq, xp, fp, left=np.nan, right=np.nan)
    vq = np.nan_to_num(vq, nan=0.0)

    # mirror to [-Lz, Lz] with odd/even symmetry
    if type_ == 1:  # odd => sin-like
        temp1 = np.concatenate([-xq[:0:-1], xq])
        temp2 = np.concatenate([-vq[:0:-1], vq])
    else:  # even => cos-like
        temp1 = np.concatenate([-xq[:0:-1], xq])
        temp2 = np.concatenate([vq[:0:-1], vq])
    return np.column_stack([temp1, temp2])


def _xcorr0_normalized(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if np.allclose(a, 0) or np.allclose(b, 0):
        return np.nan
    num = float(np.dot(a, b))
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return num / den if den != 0 else np.nan


def temp_loading_fft(load_data: np.ndarray, factor_corr: float = 0.98) -> tuple[np.ndarray, int, np.ndarray]:
    """Port of temp_Loading_FFT.m (best-effort)."""
    z = load_data[:, 0]
    load_original = load_data[:, 1]

    load = load_original.copy()
    load[1:-1] = load_original[1:-1][::-1]

    N = len(z)
    fourier = np.fft.fftshift(np.fft.fft(load))
    half = (N + 1) // 2

    F_real = np.real(fourier[half - 1:])
    F_imag = np.imag(fourier[half - 1:])
    F_entire = np.column_stack([F_real, F_imag])  # (half,2)

    cos_temp = np.abs(F_real)
    sin_temp = np.abs(F_imag)
    entire_temp = np.concatenate([cos_temp, sin_temp])

    entire_index = np.zeros(len(entire_temp), dtype=int)  # MATLAB-like 1-based
    load_weight = np.zeros(N, dtype=float)
    temp = np.zeros(len(entire_temp) + 1, dtype=float)

    i = 0
    while True:
        cc = _xcorr0_normalized(load_original, load_weight)
        if (not np.isnan(cc)) and (cc >= factor_corr - 1e-5):
            break
        i += 1
        index = int(np.argmax(entire_temp))
        entire_index[i - 1] = index + 1
        entire_temp[index] = 0.0

        for n_ in range(1, N + 1):
            for k in range(i):
                idx_k = entire_index[k]
                if idx_k <= half:
                    temp[idx_k] = np.cos(2 * np.pi * (idx_k - 1) * n_ / N)
                else:
                    temp[idx_k] = np.sin(2 * np.pi * (idx_k - half - 1) * n_ / N)

            idxs = entire_index[:i]
            vals = []
            for idx_k in idxs:
                if idx_k <= half:
                    vals.append(F_entire[idx_k - 1, 0])
                else:
                    vals.append(F_entire[idx_k - half - 1, 1])
            vals = np.asarray(vals, dtype=float)

            load_weight[n_ - 1] = (-F_real[0] + 2 * np.dot(vals, temp[idxs])) / N

        if i >= len(entire_temp) - 1:
            break

    factor = (np.max(load_original) / np.max(load_weight)) if np.max(load_weight) != 0 else 1.0
    F_entire = F_entire * factor

    entire_index = entire_index[:i]
    return entire_index, N, F_entire


def semi_circular(lengths_mm: np.ndarray, num_points: int = 100) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.asarray(lengths_mm, dtype=float).ravel()
    x = np.zeros((len(lengths), num_points), dtype=float)
    y = np.zeros((len(lengths), num_points), dtype=float)
    for i, ln in enumerate(lengths):
        xi = np.linspace(-ln, ln, num_points)
        yi = np.sqrt(np.maximum(0.0, 1.0 - (xi / ln) ** 2))
        x[i, :] = xi
        y[i, :] = yi
    return y, x


def loading(factor: np.ndarray, V_in: float) -> LoadingResult:
    """Port of Loading.m (best-effort)."""
    n = 24
    ribs = 2
    z0 = 1.0
    Lz = 8.0
    t = 0.01
    factor_corr = 0.98
    V = float(V_in)
    lengths = np.array([80.0, 120.0], dtype=float)  # mm

    loadfourier3D = np.zeros((1001, 6 * (n + 1) + 1, ribs), dtype=float)

    factor = np.asarray(factor, dtype=float).reshape(-1, 1)
    factor = np.vstack([np.ones((11, 1)), factor])

    templ = np.zeros(ribs, dtype=int)
    for i in range(ribs):
        temp = loading3d_fourier(Lz, z0, i + 1, V, t, n, factor_corr, lengths)
        loadfourier3D[:temp.shape[0], :, i] = temp
        templ[i] = temp.shape[0]
    loadfourier3D = loadfourier3D[:np.max(templ), :, :]
    return LoadingResult(loadfourier3D=loadfourier3D, Lz=Lz, n=n, t=t, V=V)


def loading3d_fourier(Lz: float, z: float, ii: int, V: float, t: float, n: int,
                     factor_corr: float, lengths: np.ndarray) -> np.ndarray:
    """Generate loadfourier3D for one rib. Only VERTICAL channels are populated correctly for solver."""
    # ---------- vertical ----------
    Fourier_entire_vertical = np.zeros((1001, 2 * (n + 1) + 1), dtype=float)
    Fourier_entire_vertical[:, 0] = np.arange(0, 1001)

    # temp buffers
    temp1 = np.zeros((int(2 * Lz / 0.01 + 2), n + 1), dtype=float)
    temp2 = np.zeros_like(temp1)

    for I in range(n + 1):
        pressure, ln = semi_circular(lengths)
        tempr = ln[ii - 1, :] / 1000.0 + z + V * t * I

        l1 = np.linspace(0, np.min(tempr), 101) if np.min(tempr) > 0 else np.array([0.0])
        l2 = np.linspace(np.max(tempr), Lz, 101) if (Lz - np.max(tempr)) > 0 else np.array([Lz])

        load_raw = np.zeros((len(l1) + len(tempr) + len(l2) - 2, 2), dtype=float)
        load_raw[:, 0] = np.concatenate([l1[:-1], tempr, l2[1:]])
        load_raw[len(l1) - 1:len(l1) - 1 + len(tempr), 1] = pressure[ii - 1, :]

        load_data = temp_interp_loading(load_raw, Lz, type_=1)
        entire_index, N, F_entire = temp_loading_fft(load_data, factor_corr=factor_corr)

        temp1[:len(entire_index), I] = entire_index
        vec = np.concatenate([F_entire[:, 0], F_entire[:, 1]])  # [real-half, imag-half]
        temp2[:len(vec), I] = vec

    # N points in [-Lz..Lz] with step 0.01
    # NOTE: our temp_loading_fft returns N from the data length; keep consistent:
    N = int(round(2 * Lz / 0.01)) + 1

    for i in range(n + 1):
        entire_index = temp1[:np.count_nonzero(temp1[:, i]), i].astype(int)
        F_ent = temp2[:, i]

        # MATLAB mapping logic (best-effort)
        half = (N + 1) // 2
        for idx in entire_index:
            if (idx - half - 1) > 0:
                L = int(idx - half - 1)
                # sin-channel
                Fourier_entire_vertical[L, 2 * i + 2] = F_ent[idx - 1]
            else:
                L = int(idx - 1)
                # cos-channel
                Fourier_entire_vertical[L, 2 * i + 1] = F_ent[idx - 1]

    mask = ~(np.all(Fourier_entire_vertical[:, 1:] == 0, axis=1))
    Fourier_entire_vertical = Fourier_entire_vertical[mask]
    Fourier_entire_vertical[:, 1:] = 2.0 / N * Fourier_entire_vertical[:, 1:]
    Fourier_entire_vertical[0, 1:] *= 0.5

    # ---------- build loadfourier3D ----------
    loadfourier3D = np.zeros((1001, 1 + 6 * (n + 1)), dtype=float)
    loadfourier3D[:, 0] = np.arange(0, 1001)

    # ✅ 关键：把 vertical 的 sin/cos 写入 MATLAB 读取的 5:6:end / 6:6:end
    # MATLAB: 5:6:end => Python index 4::6
    # MATLAB: 6:6:end => Python index 5::6
    for row in Fourier_entire_vertical:
        L = int(row[0])
        cos_series = row[1::2]  # length n+1
        sin_series = row[2::2]  # length n+1

        loadfourier3D[L, 4::6] = sin_series  # vertical sin  -> MATLAB 5:6:end
        loadfourier3D[L, 5::6] = cos_series  # vertical cos  -> MATLAB 6:6:end

    # 🚫 先不要写 longitudinal，避免污染同一个 6-block 的列
    # （等 vertical 与 solver 完全对齐后，再把 longitudinal 放到 2::6 / 3::6 等正确通道）

    mask = ~(np.all(loadfourier3D[:, 1:] == 0, axis=1))
    loadfourier3D = loadfourier3D[mask]
    return loadfourier3D


def temp_loadgeneration(GDof: int, n: int, l: int, loadfourier3D: np.ndarray,
                        Loading_index: np.ndarray, Lz: float) -> np.ndarray:
    """Port of temp_loadgeneration.m (keep MATLAB behavior)."""
    loading = np.zeros((GDof, n + 1), dtype=float)

    ntrip = Loading_index.shape[0] // 3
    ribs = loadfourier3D.shape[2]

    for i in range(ntrip):
        nodes = Loading_index[3 * i:3 * i + 3, 0].astype(int)
        w = Loading_index[3 * i:3 * i + 3, 1].astype(float)
        rib = i % ribs

        if l == 0:
            # MATLAB uses row 1 (index 0) when l==0
            loading[nodes + GDof // 3, :] = (Lz / 2) * w[:, None] * ((-1) ** l) * \
                                            loadfourier3D[0, 4::6, rib][None, :]  # sin
            loading[nodes + 2 * (GDof // 3), :] = (Lz / 2) * w[:, None] * ((-1) ** l) * \
                                                  loadfourier3D[0, 5::6, rib][None, :]  # cos
        else:
            Ls = loadfourier3D[:, 0, rib].astype(int)
            jj = np.where(Ls == int(l))[0]
            if len(jj) == 0:
                continue
            j = int(jj[0])

            loading[nodes + GDof // 3, :] = (Lz / 2) * w[:, None] * ((-1) ** l) * \
                                            loadfourier3D[j, 4::6, rib][None, :]  # sin

            # MATLAB “quirk”: cos uses row 1 always
            loading[nodes + 2 * (GDof // 3), :] = (Lz / 2) * w[:, None] * ((-1) ** l) * \
                                                  loadfourier3D[0, 5::6, rib][None, :]  # cos

    loading = -1e6 * loading  # direction and MPa->Pa
    return loading
