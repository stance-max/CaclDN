from __future__ import annotations

from typing import Any, Dict

import numpy as np

from DataManager import DataManager


def calculate_parameters(data_manager: DataManager) -> Dict[str, Any]:
    """Расчёт параметров 2D/3D ДН и пеленгационного направления.

    Считает:
    - 2D: КНД, ширина ГЛ -3дБ, УБЛ, следующий побочный максимум, СКО УБЛ,
      пеленгационное направление (суммарная/разностная для XOZ/YOZ).
    - 3D: КНД, ширина ГЛ (по срезам 3D), УБЛ, следующий побочный максимум, СКО УБЛ.

    Результаты пишутся в `calc_params` (числовые float/int) и в `results['params']`.
    """

    st = data_manager.state
    arr = st.array
    pat = st.pattern
    plots = st.plots
    ca = st.calc_arrays

    out: Dict[str, Any] = {"dn2": None, "dn3": None}

    if plots.open_2d:
        out["dn2"] = _calc_dn2_params(data_manager, arr, pat, ca, plots)

    if plots.open_3d and plots.open_3d_sum:
        out["dn3"] = _calc_dn3_params(data_manager, arr, pat, ca)

    data_manager.set_result("params", out)
    data_manager.set("params", out)
    return out


def _calc_dn2_params(data_manager: DataManager, arr: Any, pat: Any, ca: Any, plots: Any) -> Dict[str, Any]:
    ang = _arr(ca.dn2_angles_deg)
    sx_db = _arr(ca.dn2_sum_xoz_db)
    sy_db = _arr(ca.dn2_sum_yoz_db)
    rx_db = _arr(ca.dn2_diff_xoz_db)
    ry_db = _arr(ca.dn2_diff_yoz_db)

    sx_c = _arr(ca.dn2_sum_xoz_complex, complex_=True)
    sy_c = _arr(ca.dn2_sum_yoz_complex, complex_=True)

    if ang is None or sx_db is None or sy_db is None or sx_c is None or sy_c is None:
        raise ValueError("Для расчёта 2D параметров нужны результаты CalcDN2 в CalculationArrays.")

    x_axis = np.sin(np.deg2rad(ang))

    prm_x = _analyze_cut(x_axis, sx_db)
    prm_y = _analyze_cut(x_axis, sy_db)

    # КНД по формуле из CalcDN.m для множителя решётки.
    lam = float(arr.wavelength_m)
    cos0 = np.cos(np.deg2rad(float(pat.th0)))
    knd_x = 10.0 * np.log10(max(4 * np.pi * arr.dx * arr.dy / (lam**2) * np.max(np.abs(sx_c)) * max(cos0, 1e-12), 1e-15))
    knd_y = 10.0 * np.log10(max(4 * np.pi * arr.dx * arr.dy / (lam**2) * np.max(np.abs(sy_c)) * max(cos0, 1e-12), 1e-15))

    data_manager.set_calc_param("knd_x_db", float(knd_x))
    data_manager.set_calc_param("knd_y_db", float(knd_y))

    _store_cut_params(data_manager, "x", prm_x)
    _store_cut_params(data_manager, "y", prm_y)

    pel = None
    if plots.open_peling and rx_db is not None and ry_db is not None:
        pel_x = _find_peling_direction(ang, sx_db, rx_db, expected_deg=float(pat.th0), width_deg=prm_x["width_deg"])
        pel_y = _find_peling_direction(ang, sy_db, ry_db, expected_deg=float(pat.ph0), width_deg=prm_y["width_deg"])

        _store_peling(data_manager, "x", pel_x)
        _store_peling(data_manager, "y", pel_y)
        pel = {"x": pel_x, "y": pel_y}

    return {
        "knd_x_db": float(knd_x),
        "knd_y_db": float(knd_y),
        "x": prm_x,
        "y": prm_y,
        "peling": pel,
    }


def _calc_dn3_params(data_manager: DataManager, arr: Any, pat: Any, ca: Any) -> Dict[str, Any]:
    x_deg = _arr(ca.dn3_x_deg)
    y_deg = _arr(ca.dn3_y_deg)
    sum_db = _arr(ca.dn3_sum_db)
    if x_deg is None or y_deg is None or sum_db is None:
        raise ValueError("Для расчёта 3D параметров нужны dn3_x_deg/dn3_y_deg/dn3_sum_db.")

    # Нормированный линейный уровень
    lin = np.power(10.0, sum_db / 20.0)
    iy, ix = np.unravel_index(np.argmax(sum_db), sum_db.shape)

    x_cut_db = sum_db[iy, :]
    y_cut_db = sum_db[:, ix]
    x_cut_axis = x_deg[iy, :]
    y_cut_axis = y_deg[:, ix]

    width_x = _beam_width_deg_from_db(x_cut_axis, x_cut_db)
    width_y = _beam_width_deg_from_db(y_cut_axis, y_cut_db)

    main_mask = _main_lobe_component(sum_db, iy, ix, level_db=-3.0)
    side_vals = sum_db[~main_mask]
    ubl = float(np.max(side_vals)) if side_vals.size else float("nan")

    side_peaks = _local_maxima_2d(sum_db, exclude_mask=main_mask)
    side_peak_values = np.array([p[2] for p in side_peaks], dtype=float) if side_peaks else np.array([], dtype=float)
    side_peak_values = side_peak_values[side_peak_values < 0.0]
    rms_ubl = float(np.sqrt(np.mean(side_peak_values**2))) if side_peak_values.size else float("nan")
    side_next = float(np.sort(side_peak_values)[-2]) if side_peak_values.size >= 2 else float("nan")

    # КНД 3D (как в CalcDN.m):
    x = np.sin(np.deg2rad(x_deg))
    y = np.sin(np.deg2rad(y_deg))
    denom = np.sqrt(np.maximum(1.0 - x**2 - y**2, 1e-12))
    bb = (lin**2) / denom
    m = sum_db.shape[0]
    knd3 = 10.0 * np.log10(np.maximum(np.pi * (m**2) / np.sum(bb), 1e-15))

    data_manager.set_calc_param("knd_3d_db", float(knd3))
    data_manager.set_calc_param("width_3d_x_deg", float(width_x))
    data_manager.set_calc_param("width_3d_y_deg", float(width_y))
    data_manager.set_calc_param("ubl_3d_db", float(ubl))
    data_manager.set_calc_param("sidelobe_rms_3d_db", float(rms_ubl))
    data_manager.set_calc_param("sidelobe_next_3d_db", float(side_next))

    return {
        "knd_3d_db": float(knd3),
        "width_3d_x_deg": float(width_x),
        "width_3d_y_deg": float(width_y),
        "ubl_3d_db": float(ubl),
        "sidelobe_rms_3d_db": float(rms_ubl),
        "sidelobe_next_3d_db": float(side_next),
    }


def _analyze_cut(axis_sin: np.ndarray, mas_db: np.ndarray) -> Dict[str, Any]:
    ang = np.rad2deg(np.arcsin(np.clip(axis_sin, -1.0, 1.0)))

    width = _beam_width_deg_from_db(ang, mas_db)

    peaks_idx = _local_maxima_1d(mas_db)
    main_idx = int(np.argmax(mas_db))
    main_ang = float(ang[main_idx])

    cut = mas_db >= -3.0
    idx_in = np.flatnonzero(cut)
    if idx_in.size:
        i1, i2 = int(idx_in[0]), int(idx_in[-1])
    else:
        i1 = i2 = main_idx

    outside = np.r_[mas_db[:i1], mas_db[i2 + 1 :]]
    ubl = float(np.max(outside)) if outside.size else float("nan")

    # Первый максимум вне границы бокового лепестка (PBL) слева/справа.
    pbl_l = _first_peak_from_side(mas_db, side="left", start=main_idx)
    pbl_r = _first_peak_from_side(mas_db, side="right", start=main_idx)

    # Следующий побочный максимум после первого БЛ: максимум за первым пиком.
    next_left = _next_peak_after(mas_db, pbl_l[0], side="left") if pbl_l[0] >= 0 else (-1, np.nan)
    next_right = _next_peak_after(mas_db, pbl_r[0], side="right") if pbl_r[0] >= 0 else (-1, np.nan)
    side_next = np.nanmax([next_left[1], next_right[1]])

    side_peaks_vals = np.array([mas_db[i] for i in peaks_idx if i < i1 or i > i2], dtype=float)
    side_peaks_vals = side_peaks_vals[side_peaks_vals < 0.0]
    side_rms = float(np.sqrt(np.mean(side_peaks_vals**2))) if side_peaks_vals.size else float("nan")

    return {
        "width_deg": float(width),
        "ubl_db": float(ubl),
        "pbl_left_db": float(pbl_l[1]),
        "pbl_left_deg": float(ang[pbl_l[0]]) if pbl_l[0] >= 0 else float("nan"),
        "pbl_right_db": float(pbl_r[1]),
        "pbl_right_deg": float(ang[pbl_r[0]]) if pbl_r[0] >= 0 else float("nan"),
        "sidelobe_next_db": float(side_next),
        "sidelobe_rms_db": float(side_rms),
        "main_idx": int(main_idx),
        "main_deg": main_ang,
        "main_db": float(mas_db[main_idx]),
    }


def _find_peling_direction(
    ang_deg: np.ndarray,
    sum_db: np.ndarray,
    diff_db: np.ndarray,
    *,
    expected_deg: float,
    width_deg: float,
) -> Dict[str, Any]:
    # 1) выбираем главный лепесток суммарной ДН, предпочитая близость к ожидаемому направлению.
    peaks = _local_maxima_1d(sum_db)
    if not peaks:
        main_idx = int(np.argmax(sum_db))
    else:
        candidates = np.array(peaks, dtype=int)
        # score: сначала близость к expected, затем уровень.
        dist = np.abs(ang_deg[candidates] - expected_deg)
        order = np.lexsort((-sum_db[candidates], dist))
        main_idx = int(candidates[order[0]])

    main_deg = float(ang_deg[main_idx])
    main_db = float(sum_db[main_idx])

    # 2) для разностной берём минимум между ближайшими максимумами слева/справа от главного.
    diff_peaks = _local_maxima_1d(diff_db)
    left = [i for i in diff_peaks if i < main_idx]
    right = [i for i in diff_peaks if i > main_idx]

    if left and right:
        l = max(left)
        r = min(right)
        seg = diff_db[l : r + 1]
        rel = int(np.argmin(seg))
        d_idx = l + rel
    else:
        # fallback: ищем минимум в окрестности ±max(width/2, 1°)
        window = max(int(round(max(width_deg / max(np.mean(np.diff(ang_deg)), 1e-6), 2))), 2)
        a = max(main_idx - window, 0)
        b = min(main_idx + window + 1, len(diff_db))
        d_idx = a + int(np.argmin(diff_db[a:b]))

    return {
        "sum_idx": int(main_idx),
        "sum_deg": main_deg,
        "sum_db": main_db,
        "diff_idx": int(d_idx),
        "diff_deg": float(ang_deg[d_idx]),
        "diff_db": float(diff_db[d_idx]),
    }


def _store_cut_params(dm: DataManager, axis: str, prm: Dict[str, Any]) -> None:
    dm.set_calc_param(f"width_{axis}_deg", float(prm["width_deg"]))
    dm.set_calc_param(f"ubl_{axis}_db", float(prm["ubl_db"]))
    dm.set_calc_param(f"sidelobe_next_{axis}_db", float(prm["sidelobe_next_db"]))
    dm.set_calc_param(f"sidelobe_rms_{axis}_db", float(prm["sidelobe_rms_db"]))


def _store_peling(dm: DataManager, axis: str, pel: Dict[str, Any]) -> None:
    dm.set_calc_param(f"pel_sum_{axis}_idx", int(pel["sum_idx"]))
    dm.set_calc_param(f"pel_sum_{axis}_deg", float(pel["sum_deg"]))
    dm.set_calc_param(f"pel_sum_{axis}_db", float(pel["sum_db"]))
    dm.set_calc_param(f"pel_diff_{axis}_idx", int(pel["diff_idx"]))
    dm.set_calc_param(f"pel_diff_{axis}_deg", float(pel["diff_deg"]))
    dm.set_calc_param(f"pel_diff_{axis}_db", float(pel["diff_db"]))


def _beam_width_deg_from_db(angles_deg: np.ndarray, mas_db: np.ndarray) -> float:
    cut = mas_db >= -3.0
    idx = np.flatnonzero(cut)
    if idx.size < 2:
        return float("nan")
    return float(angles_deg[idx[-1]] - angles_deg[idx[0]])


def _first_peak_from_side(mas_db: np.ndarray, *, side: str, start: int) -> tuple[int, float]:
    peaks = _local_maxima_1d(mas_db)
    if side == "left":
        cand = [i for i in peaks if i < start]
        if not cand:
            return -1, float("nan")
        i = max(cand)
        return i, float(mas_db[i])
    cand = [i for i in peaks if i > start]
    if not cand:
        return -1, float("nan")
    i = min(cand)
    return i, float(mas_db[i])


def _next_peak_after(mas_db: np.ndarray, first_idx: int, *, side: str) -> tuple[int, float]:
    peaks = _local_maxima_1d(mas_db)
    if side == "left":
        cand = [i for i in peaks if i < first_idx]
        if not cand:
            return -1, float("nan")
        i = max(cand)
        return i, float(mas_db[i])
    cand = [i for i in peaks if i > first_idx]
    if not cand:
        return -1, float("nan")
    i = min(cand)
    return i, float(mas_db[i])


def _local_maxima_1d(x: np.ndarray) -> list[int]:
    if x.size < 3:
        return []
    idx = np.where((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]))[0] + 1
    return idx.tolist()


def _main_lobe_component(db: np.ndarray, iy: int, ix: int, *, level_db: float) -> np.ndarray:
    mask = db >= level_db
    h, w = mask.shape
    comp = np.zeros_like(mask, dtype=bool)
    if not mask[iy, ix]:
        return comp
    stack = [(iy, ix)]
    comp[iy, ix] = True
    while stack:
        y, x = stack.pop()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not comp[ny, nx]:
                comp[ny, nx] = True
                stack.append((ny, nx))
    return comp


def _local_maxima_2d(db: np.ndarray, *, exclude_mask: np.ndarray) -> list[tuple[int, int, float]]:
    h, w = db.shape
    peaks: list[tuple[int, int, float]] = []
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if exclude_mask[y, x]:
                continue
            v = db[y, x]
            nb = db[y - 1 : y + 2, x - 1 : x + 2]
            if v == np.max(nb) and np.count_nonzero(nb == v) == 1:
                peaks.append((y, x, float(v)))
    return peaks


def _arr(v: Any, complex_: bool = False) -> np.ndarray | None:
    if v is None:
        return None
    return np.asarray(v, dtype=np.complex128 if complex_ else float)


__all__ = ["calculate_parameters"]
