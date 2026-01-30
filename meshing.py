from __future__ import annotations

"""Meshing utilities (strict port of Meshing.m).

This module is a direct Python translation of the MATLAB code provided by the user.

Important compatibility notes:
* Node ids are **0-based** in this Python port (MATLAB is 1-based). All downstream
  modules in this repo assume 0-based node ids.
* Element connectivity is Q4 with node ordering [n1, n2, n3, n4] matching the
  MATLAB element_node definition.
* Layer boundaries are represented with `layernumber = [0, layer_element...]`
  where `layer_element` are cumulative element counts per layer (counts, so they
  remain valid under 0-based indexing).
* Interfaces: MATLAB inserts a vertical gap `d_interface` between layers (except
  between ground origin and the first layer). We reproduce the exact y-coordinate
  construction.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class Mesh:
    layernumber: np.ndarray          # (n_layers+1,) cumulative element counts
    elementdata: np.ndarray          # (n_elem, 5): [eid, n1, n2, n3, n4] (0-based node ids)
    nodecoordinate: np.ndarray       # (n_node, 2): [x, y] in meters
    interface_node: np.ndarray       # (n_layers-1, 4): [top_row_first, top_row_last, bot_row_first, bot_row_last] (0-based)
    numberelements: int
    numbernode: int
    GDof: int
    Dx: np.ndarray                   # (N, nx_elems) element sizes along x per layer
    Dy: np.ndarray                   # (N, 10) element sizes along y per layer (padded with zeros)
    Loading_index: np.ndarray        # (n_load_nodes, 2): [node_id, node_width]


def input_tire_dimension() -> tuple[np.ndarray, float, int, np.ndarray]:
    """Port of `input_tire_dimension()` in Meshing.m.

    Returns:
        dx_tire: (m,) element widths across the tire region (m)
        d_load: total load width (m)
        node_index: number of entries in `temp` in MATLAB (used for load-node range)
        node_width: tributary width per load node (m)
    """
    rib = np.array([50.0, 60.0, 60.0], dtype=float)  # mm
    gr = 10.0  # groove mm
    center = 400.0 - (2 * rib.sum() - rib[2] + 4 * gr)  # mm

    temp = np.array(
        [rib[0] / 2, rib[0] / 2, gr,
         rib[1] / 2, rib[1] / 2, gr,
         rib[2] / 2, rib[2] / 2, gr,
         rib[1] / 2, rib[1] / 2, gr,
         rib[0] / 2, rib[0] / 2],
        dtype=float
    ) / 1000.0  # m

    node_width = np.array(
        [rib[0] / 4, rib[0] / 2, rib[0] / 4,
         rib[1] / 4, rib[1] / 2, rib[1] / 4,
         rib[2] / 4, rib[2] / 2, rib[2] / 4,
         rib[1] / 4, rib[1] / 2, rib[1] / 4,
         rib[0] / 4, rib[0] / 2, rib[0] / 4],
        dtype=float
    ) / 1000.0  # m

    node_index = int(len(temp))
    dx_tire = np.concatenate(
        [temp, (center / 1000.0) * np.array([0.25, 0.25, 0.25, 0.25], dtype=float), temp]
    )
    d_load = float(dx_tire.sum())
    return dx_tire, d_load, node_index, node_width


def a_Q4mesh_Seed(width: list[float], thickness: list[float], dx_tire: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Port of `a_Q4mesh_Seed(width, thickness, dx_tire)`.

    Inputs follow MATLAB semantics:
        width = [Lx, d_load]
        thickness = [d1, d2, ..., dN]  (direction: down to up)

    Returns:
        Dx: (N, nx) element size matrix in x (same for all layers)
        Dy: (N, 10) element size matrix in y (padded with zeros)
    """
    N = len(thickness)
    Lx = float(width[0])
    d_load = float(width[1])

    # ---- form Dx ----
    dx_load = np.array([1/4, 1/4, 1/4, 1/8, 1/16, 1/32, 1/32], dtype=float) * d_load
    Lx_remain = Lx - d_load
    dx1 = np.array([1/4, 1/4, 1/4, 1/8, 1/8], dtype=float) * Lx_remain
    dx = np.concatenate([dx1, dx_load, dx_tire, dx_load[::-1], dx1[::-1]])

    Dx = np.zeros((N, len(dx)), dtype=float)
    Dx[:] = dx[None, :]

    # ---- form Dy ----
    Dy = np.zeros((N, 10), dtype=float)
    th = np.array(thickness, dtype=float)
    if N == 4:
        Dy[0, 0:5] = np.array([1/2, 1/4, 1/8, 1/16, 1/16], dtype=float) * th[0]
        Dy[1, 0:5] = (th[1] / 5.0) * np.ones(5)
        Dy[2, 0:4] = (th[2] / 4.0) * np.ones(4)
        Dy[3, 0:4] = (th[3] / 4.0) * np.ones(4)
    elif N == 5:
        Dy[0, 0:4] = np.array([1/2, 1/4, 1/8, 1/8], dtype=float) * th[0]
        Dy[1, 0:4] = np.array([1/2, 1/4, 1/8, 1/8], dtype=float) * th[1]
        Dy[2, 0:4] = (th[2] / 4.0) * np.ones(4)
        Dy[3, 0:10] = (th[3] / 10.0) * np.ones(10)
        Dy[4, 0:4] = (th[4] / 4.0) * np.ones(4)
    elif N == 6:
        Dy[0, 0:4] = np.array([1/2, 1/4, 1/8, 1/8], dtype=float) * th[0]
        Dy[1, 0:4] = np.array([1/2, 1/4, 1/8, 1/8], dtype=float) * th[1]
        Dy[2, 0:4] = np.array([1/2, 1/4, 1/8, 1/8], dtype=float) * th[2]
        Dy[3, 0:6] = (th[3] / 6.0) * np.ones(6)
        Dy[4, 0:8] = (th[4] / 8.0) * np.ones(8)
        Dy[5, 0:10] = (th[5] / 10.0) * np.ones(10)
    else:
        raise ValueError(f"Unsupported layer count N={N}. MATLAB code supports 4/5/6.")

    return Dx, Dy


def a_Q4mesh_Meshing(Dx: np.ndarray, Dy: np.ndarray, d_interface: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Port of `a_Q4mesh_Meshing(Dx, Dy, d_interface)`.

    Returns:
        layer_element: (N,) cumulative element counts per layer
        element_node: (Elements, 4) element connectivity (0-based)
        nodecoordinate: (Nodes, 2) node coordinates (m)
        interface_node: (N-1, 4) interface row node ids (0-based)
    """
    N = Dx.shape[0]
    layer_element = np.zeros(N, dtype=int)
    Elements = 0
    Nodes = 0
    interface_node = np.zeros((max(N - 1, 0), 4), dtype=int)
    layer_nodes = np.zeros(N + 1, dtype=int)

    # ---- count totals (exactly as MATLAB) ----
    for i in range(N):
        x = int(np.count_nonzero(Dx[i, :]))
        y = int(np.count_nonzero(Dy[i, :]))
        Elements += x * y
        Nodes += (x + 1) * (y + 1)
        layer_element[i] = Elements
        if i < N - 1:
            # MATLAB stores 1-based node ids: [Nodes-x, Nodes, Nodes+1, Nodes+1+x]
            # Convert to 0-based: subtract 1
            interface_node[i, :] = np.array([
                (Nodes - 1) - x,
                (Nodes - 1),
                Nodes,
                Nodes + x
            ], dtype=int)
        layer_nodes[i + 1] = Nodes

    # ---- allocate ----
    element_node = np.zeros((Elements, 4), dtype=int)
    xc = np.zeros(Nodes, dtype=float)
    yc = np.zeros(Nodes, dtype=float)

    Nt = 0  # number of nodes filled so far
    Yc = 0.0
    layer_element_temp = np.concatenate(([0], layer_element))

    # ---- meshing iteration (direction: down to up) ----
    for i in range(N):
        dx = Dx[i, Dx[i, :] != 0.0]
        dy = Dy[i, Dy[i, :] != 0.0]
        nx_e = int(len(dx))
        ny_e = int(len(dy))
        nnodex = nx_e + 1
        nnodey = ny_e + 1

        # element-node iteration
        e0 = int(layer_element_temp[i])
        n0 = int(layer_nodes[i])
        for elerow in range(ny_e):
            for elecol in range(nx_e):
                eleno = e0 + elerow * nx_e + elecol
                n1 = n0 + elerow * nnodex + elecol
                n2 = n1 + 1
                n3 = n2 + nnodex
                n4 = n1 + nnodex
                element_node[eleno, :] = (n1, n2, n3, n4)

        # node coordinate iteration
        temp_interface = float(d_interface) if i > 0 else 0.0
        dx_temp = np.concatenate(([0.0], dx))
        dy_temp = np.concatenate(([0.0], dy))
        x_cum = np.cumsum(dx_temp)
        y_cum = np.cumsum(dy_temp)
        for row in range(nnodey):
            for col in range(nnodex):
                nt = Nt + row * nnodex + col
                xc[nt] = x_cum[col]
                yc[nt] = Yc + y_cum[row] + temp_interface
        Nt = Nt + nnodey * nnodex
        Yc = float(yc[Nt - 1])

    nodecoordinate = np.column_stack([xc[:Nt], yc[:Nt]])
    return layer_element, element_node, nodecoordinate, interface_node


def meshing(thickness_in: np.ndarray, Lx: float = 2.0, d_interface: float = 0.002) -> Mesh:
    """High-level wrapper matching MATLAB `Meshing(thickness_in)` outputs.

    Args:
        thickness_in: array-like, expected length >= 3 (MATLAB uses thickness_in(1..3)).
                      Ordering in MATLAB is [t_surface, t_base, t_subbase] (commonly),
                      and they build: thickness=[2, thickness_in(3), thickness_in(2), thickness_in(1)].
        Lx: same as MATLAB (both side length along X)
        d_interface: interface gap thickness (m)
    """
    thickness_in = np.asarray(thickness_in, dtype=float).ravel()
    if thickness_in.size < 3:
        raise ValueError("thickness_in must have at least 3 entries (surface/base/subbase).")

    dx_tire, d_load, node_index, node_width = input_tire_dimension()

    # MATLAB: thickness=[2, thickness_in(3), thickness_in(2), thickness_in(1)]
    thickness = [2.0, float(thickness_in[2]), float(thickness_in[1]), float(thickness_in[0])]
    width = [float(Lx), float(d_load)]

    Dx, Dy = a_Q4mesh_Seed(width, thickness, dx_tire)
    layer_element, element_node, nodecoordinate, interface_node = a_Q4mesh_Meshing(Dx, Dy, float(d_interface))

    numbernode = int(nodecoordinate.shape[0])
    GDof = int(3 * numbernode)
    numberelements = int(element_node.shape[0])
    elementdata = np.column_stack([np.arange(numberelements, dtype=int), element_node])
    layernumber = np.concatenate(([0], layer_element.astype(int)))

    # ---- Loading index and width (dual tire) ----
    top_y = float(sum(thickness) + float(d_interface) * (len(thickness) - 1))

    temp1 = None
    temp2 = None
    for i in range(numbernode):
        x, y = nodecoordinate[i, 0], nodecoordinate[i, 1]
        if abs(x - Lx) <= 1e-5 and abs(y - top_y) <= 1e-5:
            temp1 = i
        elif abs(x - (Lx + d_load)) <= 1e-5 and abs(y - top_y) <= 1e-5:
            temp2 = i

    if temp1 is None or temp2 is None:
        # Should not happen if meshing is consistent; provide a safe fallback.
        # Nearest nodes to the targets at the top surface.
        xy = nodecoordinate
        temp1 = int(np.argmin((xy[:, 0] - Lx) ** 2 + (xy[:, 1] - top_y) ** 2))
        temp2 = int(np.argmin((xy[:, 0] - (Lx + d_load)) ** 2 + (xy[:, 1] - top_y) ** 2))

    # MATLAB (1-based): [temp1:(temp1+node_index) (temp2-node_index):temp2]
    # Python (0-based): same formula but with 0-based ids and inclusive endpoints.
    loading_index = np.concatenate([
        np.arange(temp1, temp1 + node_index + 1, dtype=int),
        np.arange(temp2 - node_index, temp2 + 1, dtype=int)
    ])
    loading_width = np.concatenate([node_width, node_width]).astype(float)
    Loading_index = np.column_stack([loading_index, loading_width])

    return Mesh(
        layernumber=layernumber,
        elementdata=elementdata,
        nodecoordinate=nodecoordinate,
        interface_node=interface_node,
        numberelements=numberelements,
        numbernode=numbernode,
        GDof=GDof,
        Dx=Dx,
        Dy=Dy,
        Loading_index=Loading_index,
    )
