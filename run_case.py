from __future__ import annotations

import argparse
import numpy as np

from .meshing import meshing
from .loading import loading  # , loading_summary
from .material import material
from .temperature import temperature_profile
from .boundary import boundary
from .interface import interface_matrix
from .solver import fourier_solution
from .matrices import output_response


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="result_case_manual.npz")
    args = ap.parse_args()

    # ==========================================================
    # 0) 手工输入区（全部写死在这里，不从 xlsx 读）
    #    单位：m, Pa, kg/m^3, s
    # ==========================================================

    # ---- 车辆/荷载相关 ----
    V = 20.0  # m/s  (TSD速度/车速)

    # ---- 结构层厚
    # thickness_in 要给 3 个： [t1, t2, t3]（从上到下的三层厚度，不含最底下 2m 的路基）
    # 对应 4 层：
    #   Layer 1: subgrade (固定 2.0 m)  -> Elastic
    #   Layer 2: thickness_in[2]        -> Elastic (Base)
    #   Layer 3: thickness_in[1]        -> Viscoelastic (VE2)
    #   Layer 4: thickness_in[0]        -> Viscoelastic (VE1)
    thickness_in = np.array([
        0.08,  # t1: top VE layer thickness (m)
        0.12,  # t2: 2nd VE layer thickness (m)
        0.25,  # t3: base elastic layer thickness (m)
    ], dtype=float)

    # ---- 弹性层参数（2 个：base & subgrade）----
    # 注意顺序要与你 material() 里定义一致（我这套 port 默认 Ee = [E_base, E_subgrade]）
    Ee = np.array([4.86e8, 1.80e8], dtype=float)  # Pa  (示例：486 MPa, 180 MPa)

    # ---- Rayleigh 阻尼（2 个：base & subgrade 或全局用同一组）----
    damping = np.array([0.0, 0.0], dtype=float)  # 先给 0，等你对齐再调

    # ---- 黏弹性层参数（两层 VE：用 Einf1/Ei_1 对应 VE1；Einf2/Ei_2 对应 VE2）----
    # Ei_*：8 项 Prony/松弛模量项（示例用相同的数组；你可以分别给两层）
    Ei_1 = np.array([7.0e9, 5.0e9, 3.0e9, 2.0e9, 1.2e9, 8.0e8, 5.0e8, 3.0e8], dtype=float)
    Ei_2 = np.array([6.0e9, 4.5e9, 2.8e9, 1.8e9, 1.0e9, 7.0e8, 4.5e8, 2.5e8], dtype=float)
    Einf1 = 4.86e8  # Pa  (VE1 平衡模量/长时模量)
    Einf2 = 4.86e8  # Pa  (VE2 平衡模量/长时模量)

    # ---- 温度输入（你原来是 2 个：air & surface）----
    temperature_in = np.array([20.0, 25.0], dtype=float)

    # ==========================================================
    # 1) 网格（严格复刻你 MATLAB Meshing.m 的 Dx/Dy/编号逻辑）
    # ==========================================================
    mesh = meshing(thickness_in, Lx=2.0, d_interface=0.002)

    # ==========================================================
    # 2) 荷载（保持你原先 port 的接口）
    # ==========================================================
    load_res = loading(factor=np.ones(10), V_in=V)
    loadfourier3D = load_res.loadfourier3D
    Lz = load_res.Lz
    n = load_res.n
    t = load_res.t

    # Fourier 模态列表（建议保留 0 模态，让 solver 自己处理）
    L = np.unique(loadfourier3D[:, 0].astype(int))
    L = np.sort(L)  # 保留 0
    print("[RUN] L used:", L[:20], "len=", len(L), "has0=", (0 in set(L.tolist())))

    # ==========================================================
    # 3) 材料 / 温度 / 边界 / 界面
    # ==========================================================
    rho, v, E, D, ray, Ei_out, Ki, Gi, T = material(
        Einf1, Ei_1,
        Einf2, Ei_2,
        Ee, damping
    )

    temperature_data = temperature_profile(temperature_in)

    Bottom, Vertical = boundary(
        E, Ei_out, rho, v,
        mesh.GDof, mesh.numbernode,
        mesh.nodecoordinate,
        mesh.Dx, mesh.Dy,
        Lz
    )

    interface = interface_matrix(mesh.interface_node, mesh.nodecoordinate, Lz)

    # ==========================================================
    # 4) 求解（Fourier + 时间域）
    # ==========================================================
    U, Vv, A = fourier_solution(
        GDof=mesh.GDof,
        elementdata=mesh.elementdata,
        numbernodes=mesh.numbernode,
        nodecoordinate=mesh.nodecoordinate,
        matD=D,
        Lz=Lz,
        rho=rho,
        ray=ray,
        Ki=Ki,
        Gi=Gi,
        T=T,
        layernumber=mesh.layernumber,
        loadfourier3D=loadfourier3D,
        Bottom=Bottom,
        Vertical=Vertical,
        interface=interface,
        L=L,
        n=n,
        t=t,
        Loading_index=mesh.Loading_index,
        temperature_data=temperature_data
    )

    # ==========================================================
    # 5) 输出物理域响应（z=0 表面；你要传感器深度就改 z）
    # ==========================================================
    disp_phys, vel_phys, acc_phys = output_response(
        mesh.GDof, n,
        z=0.0, Lz=Lz, L=L,
        Displacement=U, Velocity=Vv, Acceleration=A
    )

    np.savez(
        args.out,
        nodecoordinate=mesh.nodecoordinate,
        elementdata=mesh.elementdata,
        layernumber=mesh.layernumber,
        interface_node=mesh.interface_node,
        Loading_index=mesh.Loading_index,
        loadfourier3D=loadfourier3D,
        L=L, Lz=Lz, n=n, t=t,
        Displacement=U, Velocity=Vv, Acceleration=A,
        displacement=disp_phys, velocity=vel_phys, acceleration=acc_phys,
        thickness_in=thickness_in,
        V=V, Ee=Ee, damping=damping,
        Einf1=Einf1, Einf2=Einf2, Ei_1=Ei_1, Ei_2=Ei_2,
        temperature_in=temperature_in
    )
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
