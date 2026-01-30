from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def _infer_dt(data: dict) -> float:
    """Best effort to infer dt (seconds) from saved npz."""
    if "t" in data:
        try:
            return float(np.asarray(data["t"]).ravel()[0])
        except Exception:
            pass
    # fallback
    return 0.01


def _make_time_vector(n: int, dt: float) -> np.ndarray:
    return np.arange(n + 1, dtype=float) * dt


def _find_surface_nodes(nodecoordinate: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Nodes at top surface y = max(y)."""
    y = nodecoordinate[:, 1]
    ymax = float(np.max(y))
    return np.where(np.abs(y - ymax) <= tol)[0]


def _node_dofs(node_id: int, numbernodes: int) -> tuple[int, int, int]:
    """Return (ux_dof, uy_dof, uz_dof) indices in displacement vector."""
    ux = node_id
    uy = node_id + numbernodes
    uz = node_id + 2 * numbernodes
    return ux, uy, uz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, default="case_manual.npz", help="Path to result npz")
    ap.add_argument("--node", type=int, default=None, help="Node id (0-based) to plot time histories")
    ap.add_argument("--step", type=int, default=None, help="Time step index for scatter plot (default last)")
    ap.add_argument("--save", action="store_true", help="Save png figures next to npz")
    args = ap.parse_args()

    npz_path = Path(args.npz)
    data = np.load(npz_path, allow_pickle=True)
    keys = list(data.keys())
    print("Keys in npz:", keys)

    nodecoordinate = np.asarray(data["nodecoordinate"], dtype=float)
    n = int(np.asarray(data["n"]).ravel()[0])
    dt = _infer_dt(data)
    tvec = _make_time_vector(n, dt)

    # physical response (already summed over Fourier modes at z=0 in run_case.py)
    disp = np.asarray(data["displacement"], dtype=float)  # (GDof, n+1)
    vel  = np.asarray(data["velocity"], dtype=float)
    acc  = np.asarray(data["acceleration"], dtype=float)

    GDof = disp.shape[0]
    numbernodes = GDof // 3

    # decide which node to plot
    if args.node is None:
        # pick a surface node closest to mid x
        surf = _find_surface_nodes(nodecoordinate, tol=1e-6)
        xs = nodecoordinate[surf, 0]
        xmid = 0.5 * (xs.min() + xs.max())
        node_id = int(surf[np.argmin(np.abs(xs - xmid))])
        print(f"[INFO] --node not given, auto pick surface mid node = {node_id}")
    else:
        node_id = int(args.node)

    ux_dof, uy_dof, uz_dof = _node_dofs(node_id, numbernodes)

    # ---- Figure 1: ux/uy/uz time histories ----
    plt.figure()
    plt.plot(tvec, disp[ux_dof, :], label="ux")
    plt.plot(tvec, disp[uy_dof, :], label="uy")
    plt.plot(tvec, disp[uz_dof, :], label="uz")
    plt.xlabel("t (s)")
    plt.ylabel("displacement (m)")
    plt.title(f"Displacement time history @ node {node_id}")
    plt.grid(True)
    plt.legend()

    if args.save:
        out1 = npz_path.with_suffix("").as_posix() + f"_node{node_id}_disp.png"
        plt.savefig(out1, dpi=200)

    # ---- Figure 2: uz(t) overlay of several load/surface nodes ----
    # try use Loading_index if exists; otherwise just pick a few surface nodes
    pick_nodes = None
    if "Loading_index" in data:
        Loading_index = np.asarray(data["Loading_index"], dtype=float)
        ln = Loading_index[:, 0].astype(int)
        # Loading_index contains the top-surface load nodes (MATLAB style)
        pick_nodes = ln[: min(len(ln), 8)]
        pick_nodes = np.unique(pick_nodes)
        pick_nodes = pick_nodes[pick_nodes >= 0]
        pick_nodes = pick_nodes[pick_nodes < numbernodes]
        print("[INFO] overlay nodes from Loading_index:", pick_nodes.tolist())

    if pick_nodes is None or len(pick_nodes) == 0:
        surf = _find_surface_nodes(nodecoordinate, tol=1e-6)
        pick_nodes = surf[: min(len(surf), 8)]
        print("[INFO] overlay nodes from surface nodes:", pick_nodes.tolist())

    plt.figure()
    for nid in pick_nodes:
        uz = disp[nid + 2 * numbernodes, :]
        plt.plot(tvec, uz, label=f"n{nid}")
    plt.xlabel("t (s)")
    plt.ylabel("uz (m)")
    plt.title("uz(t) overlay of several surface/load nodes")
    plt.grid(True)
    plt.legend()

    if args.save:
        out2 = npz_path.with_suffix("").as_posix() + "_uz_overlay.png"
        plt.savefig(out2, dpi=200)

    # ---- Figure 3: surface uz scatter at a given step (optional) ----
    it = (n if args.step is None else int(args.step))
    it = max(0, min(n, it))
    surf = _find_surface_nodes(nodecoordinate, tol=1e-6)
    xs = nodecoordinate[surf, 0]
    ys = nodecoordinate[surf, 1]
    uz_surf = disp[surf + 2 * numbernodes, it]

    plt.figure()
    sc = plt.scatter(xs, ys, c=uz_surf, s=15)
    plt.colorbar(sc, label="uz (m)")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(f"Surface uz scatter @ step={it}  (t={tvec[it]:.4f}s)")
    plt.grid(True)

    if args.save:
        out3 = npz_path.with_suffix("").as_posix() + f"_surf_scatter_step{it}.png"
        plt.savefig(out3, dpi=200)

    plt.show()


if __name__ == "__main__":
    main()
