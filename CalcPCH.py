from __future__ import annotations

from typing import Any, Dict

import numpy as np

from DataManager import DataManager


def calculate_pch(data_manager: DataManager) -> Dict[str, Any]:
    """Расчёт пеленгационной характеристики (CalcPH.m) по флагу open_pch."""
    st = data_manager.state
    pat = st.pattern
    plots = st.plots
    ca = st.calc_arrays

    if not plots.open_pch:
        res = {"enabled": False}
        data_manager.set_result("pch", res)
        return res

    # Нужны результаты DN2: суммарные/разностные отклики и dB.
    sx_db = _arr(ca.dn2_sum_xoz_db)
    sy_db = _arr(ca.dn2_sum_yoz_db)
    sx = _arr(ca.dn2_sum_xoz_complex, complex_=True)
    sy = _arr(ca.dn2_sum_yoz_complex, complex_=True)
    rx = _arr(ca.dn2_diff_xoz_complex, complex_=True)
    ry = _arr(ca.dn2_diff_yoz_complex, complex_=True)
    th = _arr(ca.dn2_angles_deg)

    for name, val in {
        "dn2_sum_xoz_db": sx_db,
        "dn2_sum_yoz_db": sy_db,
        "dn2_sum_xoz_complex": sx,
        "dn2_sum_yoz_complex": sy,
        "dn2_diff_xoz_complex": rx,
        "dn2_diff_yoz_complex": ry,
        "dn2_angles_deg": th,
    }.items():
        if val is None:
            raise ValueError(f"Для CalcPCH отсутствует {name} в CalculationArrays.")

    fi_n0 = float(pat.th0)
    fi_v0 = float(pat.ph0)
    xx = np.sin(np.deg2rad(th))
    yy = np.sin(np.deg2rad(th))

    # ---- Наклонная ПХ ----
    fi_n_deg, phx, krutx0, delta_phx = _calc_one(
        sum_db=sx_db,
        sum_complex=sx,
        diff_complex=rx,
        axis_sin=xx,
        axis_offset_deg=fi_n0,
    )

    # ---- Вертикальная ПХ ----
    fi_v_deg, phy, kruty0, delta_phy = _calc_one(
        sum_db=sy_db,
        sum_complex=sy,
        diff_complex=ry,
        axis_sin=yy,
        axis_offset_deg=fi_v0,
    )

    # Сохранение ключевых массивов
    data_manager.set_calc_array("pch_fi_n_deg", fi_n_deg)
    data_manager.set_calc_array("pch_nakl", phx)
    data_manager.set_calc_array("pch_fi_v_deg", fi_v_deg)
    data_manager.set_calc_array("pch_vert", phy)

    res = {
        "enabled": True,
        "krut_phx": krutx0,
        "krut_phy": kruty0,
        "delta_phx": delta_phx,
        "delta_phy": delta_phy,
        "points_n": int(fi_n_deg.size),
        "points_v": int(fi_v_deg.size),
    }
    data_manager.set_result("pch", res)
    data_manager.set("pch", res)
    return res


def _calc_one(
    *,
    sum_db: np.ndarray,
    sum_complex: np.ndarray,
    diff_complex: np.ndarray,
    axis_sin: np.ndarray,
    axis_offset_deg: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    cut = (np.sign(sum_db + 3.0) + 1.0) / 2.0
    idx = np.flatnonzero(cut == 1)
    if idx.size == 0:
        return np.array([]), np.array([]), float("nan"), float("nan")

    i1 = int(idx[0])
    i2 = int(idx[-1])

    ph = np.sin(np.angle(diff_complex) - np.angle(sum_complex)) * np.abs(diff_complex / np.maximum(sum_complex, 1e-15))
    ph_cut = ph[i1 : i2 + 1]

    fi = np.rad2deg(np.arcsin(np.clip(axis_sin[i1 : i2 + 1], -1.0, 1.0))) - axis_offset_deg

    # delta: как в MATLAB через round(...,2)==0
    zero_idx = np.flatnonzero(np.round(ph_cut, 2) == 0)
    delta = float(fi[zero_idx[0]]) if zero_idx.size else float("nan")

    # крутизна: точки на ±1/4 ширины
    width = i2 - i1
    iout = int(round(width / 4.0))
    i3 = min(max(i1 + iout, 0), len(ph) - 1)
    i4 = min(max(i2 - iout, 0), len(ph) - 1)
    x3 = np.rad2deg(np.arcsin(np.clip(axis_sin[i3], -1.0, 1.0)))
    x4 = np.rad2deg(np.arcsin(np.clip(axis_sin[i4], -1.0, 1.0)))
    if abs(x4 - x3) < 1e-12:
        slope = float("nan")
    else:
        slope = float((ph[i4] - ph[i3]) / (x4 - x3))

    return fi, ph_cut, slope, delta


def _arr(v: Any, complex_: bool = False) -> np.ndarray | None:
    if v is None:
        return None
    return np.asarray(v, dtype=np.complex128 if complex_ else float)


__all__ = ["calculate_pch"]
