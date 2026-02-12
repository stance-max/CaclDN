from __future__ import annotations

from typing import Any, Dict

import numpy as np

from DataManager import DataManager


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
        sx_db = _to_db_norm(np.abs(sx))
        sy_db = _to_db_norm(np.abs(sy))
        sum_result = {
            "xoz_linear": np.abs(sx),
            "yoz_linear": np.abs(sy),
            "xoz_db": sx_db,
            "yoz_db": sy_db,
        }

    if calc_diff and rx is not None and ry is not None and sx is not None and sy is not None:
        # Нормируем разностные относительно максимумов суммарных (как в CalcDN.m).
        sx_max = float(np.max(np.abs(sx))) if np.max(np.abs(sx)) > 0 else 1.0
        sy_max = float(np.max(np.abs(sy))) if np.max(np.abs(sy)) > 0 else 1.0
        rx_db = _to_db(np.abs(rx) / sx_max)
        ry_db = _to_db(np.abs(ry) / sy_max)
        diff_result = {
            "xoz_linear": np.abs(rx),
            "yoz_linear": np.abs(ry),
            "xoz_db": rx_db,
            "yoz_db": ry_db,
        }

    result = {
        "angles_deg": th,
        "angle_range_deg": {"th_min": th_min, "th_max": th_max, "step": step},
        "sum": sum_result,
        "diff": diff_result,
    }

    data_manager.set_result("dn2_sections", result)
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
