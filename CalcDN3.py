from __future__ import annotations

from typing import Any, Dict

import numpy as np

from DataManager import DataManager
from ImportParameters import calculate_scan_pattern_db


def calculate_dn3(data_manager: DataManager, *, chunk_size: int = 4096) -> Dict[str, Any]:
    """Расчёт 3D ДН: суммарная / азимутальная / угломестная.

    Перенос веток CalcDN.m:
    - Сумм 3D
    - Азимут 3D
    - Углов 3D

    Выполнение веток управляется флагами:
    - plots.open_3d
    - plots.open_3d_sum
    - plots.open_3d_rz_az
    - plots.open_3d_rz_el
    """

    st = data_manager.state
    arr = st.array
    pat = st.pattern
    plots = st.plots
    ca = st.calc_arrays

    if not plots.open_3d:
        data_manager.set_calc_param("dn3_enabled", False)
        data_manager.set_result("dn3", {"enabled": False})
        return {"enabled": False}

    required = [ca.amplitude, ca.xkord, ca.ykord, ca.phase_rad]
    if any(v is None for v in required):
        raise ValueError("Для CalcDN3 нужны amplitude/xkord/ykord/phase_rad в calc_arrays.")

    am = np.asarray(ca.amplitude, dtype=float)
    xkord = np.asarray(ca.xkord, dtype=float)
    ykord = np.asarray(ca.ykord, dtype=float)
    phase = np.asarray(ca.phase_rad, dtype=float)
    phase_rx = np.asarray(ca.phase_rx_rad if ca.phase_rx_rad is not None else phase, dtype=float)
    phase_ry = np.asarray(ca.phase_ry_rad if ca.phase_ry_rad is not None else phase, dtype=float)

    step3d = float(pat.step_3d)
    if step3d <= 0:
        raise ValueError("step_3d должен быть > 0")

    m = int(round((360.0 / np.pi) / step3d) + 1)
    axis = np.linspace(-1.0, 1.0, m)
    x, y = np.meshgrid(axis, axis)

    x_deg = np.rad2deg(np.arcsin(np.clip(x, -1.0, 1.0)))
    y_deg = np.rad2deg(np.arcsin(np.clip(y, -1.0, 1.0)))

    use_scan = bool(pat.use_scanpattern)
    k3d = np.ones_like(x, dtype=float)
    if use_scan:
        ds3d = np.asarray(calculate_scan_pattern_db(data_manager, x_deg, y_deg, "DS3d"), dtype=float)
        k3d = np.power(10.0, ds3d / 20.0)

    k = float(arr.wave_k)
    query = np.column_stack((x.ravel(), y.ravel()))
    x_flat = xkord.ravel()
    y_flat = ykord.ravel()

    w_sum = (am * np.exp(-1j * phase)).ravel()
    w_rx = (am * np.exp(-1j * phase_rx)).ravel()
    w_ry = (am * np.exp(-1j * phase_ry)).ravel()

    sum_db = rz_az_db = rz_el_db = None
    s_lin_2d = None

    if plots.open_3d_sum or plots.open_3d_rz_az or plots.open_3d_rz_el:
        s_lin = _beamform_points(query, x_flat, y_flat, w_sum, k, chunk_size=chunk_size)
        s_lin_2d = np.abs(s_lin).reshape(x.shape)

    if plots.open_3d_sum and s_lin_2d is not None:
        base = s_lin_2d * k3d if use_scan else s_lin_2d
        sum_db = _to_db(base / max(float(np.max(base)), 1e-15))

    norm_base = None
    if s_lin_2d is not None:
        base = s_lin_2d * k3d if use_scan else s_lin_2d
        norm_base = max(float(np.max(base)), 1e-15)

    if plots.open_3d_rz_az and norm_base is not None:
        rz_lin = np.abs(_beamform_points(query, x_flat, y_flat, w_rx, k, chunk_size=chunk_size)).reshape(x.shape)
        rz_use = rz_lin * k3d if use_scan else rz_lin
        rz_az_db = _to_db(rz_use / norm_base)

    if plots.open_3d_rz_el and norm_base is not None:
        rz_lin = np.abs(_beamform_points(query, x_flat, y_flat, w_ry, k, chunk_size=chunk_size)).reshape(x.shape)
        rz_use = rz_lin * k3d if use_scan else rz_lin
        rz_el_db = _to_db(rz_use / norm_base)

    # Метаданные в CalculationScalars
    data_manager.set_calc_param("dn3_enabled", True)
    data_manager.set_calc_param("dn3_use_scanpattern", bool(use_scan))
    data_manager.set_calc_param("dn3_step_3d", float(step3d))
    data_manager.set_calc_param("dn3_grid_m", int(m))
    data_manager.set_calc_param("dn3_grid_n", int(m))
    data_manager.set_calc_param("dn3_sum_ready", bool(sum_db is not None))
    data_manager.set_calc_param("dn3_rz_az_ready", bool(rz_az_db is not None))
    data_manager.set_calc_param("dn3_rz_el_ready", bool(rz_el_db is not None))

    # Сохранение ключевых массивов в CalculationArrays
    data_manager.set_calc_array("dn3_x_deg", x_deg)
    data_manager.set_calc_array("dn3_y_deg", y_deg)
    data_manager.set_calc_array("dn3_sum_db", sum_db)
    data_manager.set_calc_array("dn3_rz_az_db", rz_az_db)
    data_manager.set_calc_array("dn3_rz_el_db", rz_el_db)

    result = {
        "enabled": True,
        "use_scanpattern": use_scan,
        "step_3d": step3d,
        "grid_size": [int(m), int(m)],
        "sum_ready": sum_db is not None,
        "rz_az_ready": rz_az_db is not None,
        "rz_el_ready": rz_el_db is not None,
    }
    data_manager.set_result("dn3", result)
    data_manager.set("dn3", result)

    return result


def _beamform_points(
    query: np.ndarray,
    x_flat: np.ndarray,
    y_flat: np.ndarray,
    weights: np.ndarray,
    k: float,
    *,
    chunk_size: int,
) -> np.ndarray:
    out = np.zeros(query.shape[0], dtype=np.complex128)
    for i in range(0, query.shape[0], chunk_size):
        q = query[i : i + chunk_size]
        phase = -1j * k * (q[:, 0][:, None] * x_flat[None, :] + q[:, 1][:, None] * y_flat[None, :])
        out[i : i + chunk_size] = np.exp(phase) @ weights
    return out


def _to_db(v: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(v, 1e-15))


__all__ = ["calculate_dn3"]
