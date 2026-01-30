from __future__ import annotations

import numpy as np


def temperature_profile(temperature_in: np.ndarray) -> np.ndarray:
    """Port of Temperature.m.

    Returns array (N,2): [depth(m), Temp(C)] for depth 0.001..0.29 m
    """
    Tair = float(temperature_in[0])
    Tsurface = float(temperature_in[1])
    depth = np.arange(0.001, 0.29 + 1e-12, 0.001)
    out = np.zeros((len(depth), 2), dtype=float)
    out[:, 0] = depth
    # fixed hour=14 in MATLAB expression
    hour = 14.0
    term_sin1 = np.sin((hour - 15.5) * 2 * np.pi / 18)
    term_sin2 = np.sin((hour - 13.5) * 2 * np.pi / 18)
    out[:, 1] = (0.95 + 0.892 * Tsurface
                 + (np.log(depth * 1000.0) - 1.25)
                 * (-0.448 * Tsurface + 0.621 * Tair + 1.83 * term_sin1 + 0.042 * Tsurface * term_sin2))
    return out


def temperature_shift(temperature_data: np.ndarray, depth: float) -> float:
    """Approximate shift factor from temperature_data at a given depth.

    MATLAB helper `temperature(temperature_data, depth)` uses interpolation and a shift equation.
    The exact formula is embedded in MATLAB files; here we linearly interpolate and map
    to an Arrhenius-like factor around 20C as a reasonable placeholder.
    """
    temperature_data = np.asarray(temperature_data, dtype=float)
    d = temperature_data[:, 0]
    T = temperature_data[:, 1]
    Tz = float(np.interp(depth, d, T, left=T[0], right=T[-1]))
    # simple WLF-like shift relative to 20C
    C1, C2 = 8.86, 101.6
    return 10 ** (-C1 * (Tz - 20.0) / (C2 + (Tz - 20.0)))
