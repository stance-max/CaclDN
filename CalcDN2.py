from __future__ import annotations

from typing import Any, Dict

import numpy as np

from DataManager import DataManager
from ImportParameters import calculate_scan_pattern_db


def calculate_dn2_sections(
    data_manager: DataManager,
    *,
    calc_sum: bool = True,
    calc_diff: bool = True,
    chunk_size: int = 128,
) -> Dict[str, Any]:
    """Расчёт 2D-сечений ДН (XOZ/YOZ) по мотивам CalcDN.m.

    Оптимизация:
    - расчёт векторизован по элементам решётки;
    - расчёт по углам выполняется чанками, чтобы ограничить память.

    Пока не рассчитываются ПХ/пеленгация/параметры ширины и т.п.
    """

    state = data_manager.state
    arr = state.array
    pat = state.pattern
    ca = state.calc_arrays

    if ca.amplitude is None or ca.xkord is None or ca.ykord is None or ca.phase_rad is None:
        raise ValueError("Для расчёта DN2 требуются amplitude/xkord/ykord/phase_rad в calc_arrays.")

    am = np.asarray(ca.amplitude, dtype=float)
    xkord = np.asarray(ca.xkord, dtype=float)
    ykord = np.asarray(ca.ykord, dtype=float)
    phase = np.asarray(ca.phase_rad, dtype=float)

    phase_rx = np.asarray(ca.phase_rx_rad if ca.phase_rx_rad is not None else phase, dtype=float)
    phase_ry = np.asarray(ca.phase_ry_rad if ca.phase_ry_rad is not None else phase, dtype=float)

    if am.shape != xkord.shape or am.shape != ykord.shape or am.shape != phase.shape:
        raise ValueError("Размеры amplitude/xkord/ykord/phase должны совпадать.")

    th_min = float(pat.th_min)
    th_max = float(pat.th_max)
    step = float(pat.step_2d)
    if step <= 0:
        raise ValueError("step_2d должен быть > 0")

    th = np.arange(th_min, th_max + step / 2.0, step, dtype=float)
    x_axis = np.sin(np.deg2rad(th))
    y_axis = np.sin(np.deg2rad(th))

    xx_const = np.sin(np.deg2rad(float(pat.th0)))
    yy_const = np.sin(np.deg2rad(float(pat.ph0)))

    # Режим расчёта 2D-сечений: множитель решётки / с учётом ДС.
    use_scan_pattern = bool(pat.use_scanpattern)
    dskx = np.ones_like(th, dtype=float)
    dsky = np.ones_like(th, dtype=float)
    if use_scan_pattern:
        dsx = np.asarray(calculate_scan_pattern_db(data_manager, th, th, "DSxoz"), dtype=float)
        dsy = np.asarray(calculate_scan_pattern_db(data_manager, th, th, "DSyoz"), dtype=float)
        dskx = np.power(10.0, dsx / 20.0)
        dsky = np.power(10.0, dsy / 20.0)

    k = float(arr.wave_k)

    # Предвычисление комплексных весов (уменьшает вычисления в циклах по углам).
    w_sum = (am * np.exp(-1j * phase)).ravel()
    w_rx = (am * np.exp(-1j * phase_rx)).ravel()
    w_ry = (am * np.exp(-1j * phase_ry)).ravel()
    x_flat = xkord.ravel()
    y_flat = ykord.ravel()

    sx = rx = sy = ry = None
    if calc_sum or calc_diff:
        sx = _calc_xoz_response(x_axis, yy_const, k, x_flat, y_flat, w_sum, chunk_size=chunk_size)
        sy = _calc_yoz_response(y_axis, xx_const, k, x_flat, y_flat, w_sum, chunk_size=chunk_size)

    if calc_diff:
        rx = _calc_xoz_response(x_axis, yy_const, k, x_flat, y_flat, w_rx, chunk_size=chunk_size)
        ry = _calc_yoz_response(y_axis, xx_const, k, x_flat, y_flat, w_ry, chunk_size=chunk_size)

    sum_result: Dict[str, Any] | None = None
    diff_result: Dict[str, Any] | None = None

    if calc_sum and sx is not None and sy is not None:
        sx_abs = np.abs(sx)
        sy_abs = np.abs(sy)

        i_max_x = int(np.argmax(sx_abs)) if sx_abs.size else 0
        i_max_y = int(np.argmax(sy_abs)) if sy_abs.size else 0

        if use_scan_pattern:
            max_x = sx_abs[i_max_x] * dskx[i_max_x] if sx_abs.size else 1.0
            max_y = sy_abs[i_max_y] * dsky[i_max_y] if sy_abs.size else 1.0
            sx_norm = (sx_abs * dskx) / max(max_x, 1e-15)
            sy_norm = (sy_abs * dsky) / max(max_y, 1e-15)
        else:
            sx_norm = sx_abs / max(float(np.max(sx_abs)), 1e-15)
            sy_norm = sy_abs / max(float(np.max(sy_abs)), 1e-15)

        sx_db = _to_db(sx_norm)
        sy_db = _to_db(sy_norm)
        sum_result = {
            "xoz_linear": sx_abs,
            "yoz_linear": sy_abs,
            "xoz_db": sx_db,
            "yoz_db": sy_db,
        }

    if calc_diff and rx is not None and ry is not None and sx is not None and sy is not None:
        sx_abs = np.abs(sx)
        sy_abs = np.abs(sy)
        rx_abs = np.abs(rx)
        ry_abs = np.abs(ry)

        i_max_x = int(np.argmax(sx_abs)) if sx_abs.size else 0
        i_max_y = int(np.argmax(sy_abs)) if sy_abs.size else 0

        if use_scan_pattern:
            max_x = sx_abs[i_max_x] * dskx[i_max_x] if sx_abs.size else 1.0
            max_y = sy_abs[i_max_y] * dsky[i_max_y] if sy_abs.size else 1.0
            rx_norm = (rx_abs * dskx) / max(max_x, 1e-15)
            ry_norm = (ry_abs * dsky) / max(max_y, 1e-15)
        else:
            sx_max = max(float(np.max(sx_abs)), 1e-15)
            sy_max = max(float(np.max(sy_abs)), 1e-15)
            rx_norm = rx_abs / sx_max
            ry_norm = ry_abs / sy_max

        rx_db = _to_db(rx_norm)
        ry_db = _to_db(ry_norm)
        diff_result = {
            "xoz_linear": rx_abs,
            "yoz_linear": ry_abs,
            "xoz_db": rx_db,
            "yoz_db": ry_db,
        }

    result = {
        "angles_deg": th,
        "angle_range_deg": {"th_min": th_min, "th_max": th_max, "step": step},
        "use_scanpattern": use_scan_pattern,
        "sum": sum_result,
        "diff": diff_result,
    }

    # Метаданные в CalculationScalars.
    data_manager.set_calc_param("dn2_points", int(th.size))
    data_manager.set_calc_param("dn2_use_scanpattern", bool(use_scan_pattern))
    data_manager.set_calc_param("dn2_sum_ready", bool(sum_result is not None))
    data_manager.set_calc_param("dn2_diff_ready", bool(diff_result is not None))

    # Ключевые массивы сохраняем в CalculationArrays.
    data_manager.set_calc_array("dn2_angles_deg", th)

    data_manager.set_calc_array("dn2_sum_xoz_complex", None if sx is None else sx)
    data_manager.set_calc_array("dn2_sum_yoz_complex", None if sy is None else sy)
    data_manager.set_calc_array("dn2_diff_xoz_complex", None if rx is None else rx)
    data_manager.set_calc_array("dn2_diff_yoz_complex", None if ry is None else ry)

    data_manager.set_calc_array("dn2_sum_xoz_linear", None if sum_result is None else sum_result["xoz_linear"])
    data_manager.set_calc_array("dn2_sum_yoz_linear", None if sum_result is None else sum_result["yoz_linear"])
    data_manager.set_calc_array("dn2_sum_xoz_db", None if sum_result is None else sum_result["xoz_db"])
    data_manager.set_calc_array("dn2_sum_yoz_db", None if sum_result is None else sum_result["yoz_db"])

    data_manager.set_calc_array("dn2_diff_xoz_linear", None if diff_result is None else diff_result["xoz_linear"])
    data_manager.set_calc_array("dn2_diff_yoz_linear", None if diff_result is None else diff_result["yoz_linear"])
    data_manager.set_calc_array("dn2_diff_xoz_db", None if diff_result is None else diff_result["xoz_db"])
    data_manager.set_calc_array("dn2_diff_yoz_db", None if diff_result is None else diff_result["yoz_db"])

    # В results храним метаданные диапазона/режима.
    data_manager.set_result(
        "dn2_sections",
        {
            "angle_range_deg": result["angle_range_deg"],
            "use_scanpattern": use_scan_pattern,
            "sum_ready": sum_result is not None,
            "diff_ready": diff_result is not None,
            "points": int(th.size),
        },
    )
    data_manager.set(
        "dn2_sections",
        {
            "angle_range_deg": result["angle_range_deg"],
            "sum_ready": sum_result is not None,
            "diff_ready": diff_result is not None,
            "points": int(th.size),
        },
    )

    return result


def _calc_xoz_response(
    x_axis: np.ndarray,
    yy_const: float,
    k: float,
    x_flat: np.ndarray,
    y_flat: np.ndarray,
    weights: np.ndarray,
    *,
    chunk_size: int,
) -> np.ndarray:
    out = np.zeros(x_axis.size, dtype=np.complex128)
    base_y = yy_const * y_flat
    for i in range(0, x_axis.size, chunk_size):
        xs = x_axis[i : i + chunk_size]
        phase = -1j * k * (xs[:, None] * x_flat[None, :] + base_y[None, :])
        out[i : i + chunk_size] = np.exp(phase) @ weights
    return out


def _calc_yoz_response(
    y_axis: np.ndarray,
    xx_const: float,
    k: float,
    x_flat: np.ndarray,
    y_flat: np.ndarray,
    weights: np.ndarray,
    *,
    chunk_size: int,
) -> np.ndarray:
    out = np.zeros(y_axis.size, dtype=np.complex128)
    base_x = xx_const * x_flat
    for i in range(0, y_axis.size, chunk_size):
        ys = y_axis[i : i + chunk_size]
        phase = -1j * k * (base_x[None, :] + ys[:, None] * y_flat[None, :])
        out[i : i + chunk_size] = np.exp(phase) @ weights
    return out


def _to_db_norm(values: np.ndarray) -> np.ndarray:
    vmax = float(np.max(values)) if values.size > 0 else 0.0
    if vmax <= 0:
        return np.full(values.shape, -300.0, dtype=float)
    return _to_db(values / vmax)


def _to_db(values: np.ndarray) -> np.ndarray:
    eps = 1e-15
    return 20.0 * np.log10(np.maximum(values, eps))


__all__ = ["calculate_dn2_sections"]
