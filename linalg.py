from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def solve_linear_system(A: sp.spmatrix | np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve A x = b.

    MATLAB code uses backslash and (sometimes) GPU. Here we use sparse direct solve
    when possible.
    """
    if sp.issparse(A):
        return spla.spsolve(A.tocsr(), b)
    return np.linalg.solve(A, b)
