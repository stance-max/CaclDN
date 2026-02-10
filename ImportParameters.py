from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

from DataManager import DataManager, ScanPatternType

try:
    from scipy.interpolate import griddata as _scipy_griddata
except Exception:  # pragma: no cover - optional dependency
    _scipy_griddata = None

_MIDDLE_LIT = {39, 41, 51, 53, 63}
_LOWER_LIT = {1, 27, 29}
_UPPER_LIT = {65, 75, 77, 101}

_DEFAULT_FILE_BY_TYPE: Dict[ScanPatternType, str] = {
    ScanPatternType.MIDDLE: "DS_s.txt",
    ScanPatternType.LOWER: "DS_n.txt",
    ScanPatternType.UPPER: "DS_v.txt",
}


# ------------------------------ Public API -------------------------------
def import_scan_pattern(data_manager: DataManager, *, base_dir: str | Path = ".") -> np.ndarray:
    """Импорт таблицы диаграммы сканирования (DNscan.m)."""
    state = data_manager.state
    pattern = state.pattern
    arr = state.array

    scan_type = _resolve_scan_pattern_type(requested=pattern.scan_pattern_type, liter=int(arr.liter))
    file_path = _resolve_scan_file_path(
        base_dir=base_dir,
        requested_path=pattern.scan_pattern_file_path,
        scan_type=scan_type,
    )
    table = _load_scan_table(file_path)

    data_manager.patch_state(
        "pattern",
        {
            "scan_pattern_loaded": True,
            "scan_pattern_file_path": str(file_path),
            "scan_pattern_type": scan_type,
        },
    )
    data_manager.set_calc_array("scan_pattern_raw", table)
    data_manager.set_result(
        "scan_pattern",
        {
            "loaded": True,
            "scan_pattern_type": scan_type.value,
            "file_path": str(file_path),
            "rows": int(table.shape[0]),
            "columns": int(table.shape[1]),
        },
    )
    return table


def calculate_scan_pattern_db(
    data_manager: DataManager,
    xx: np.ndarray,
    yy: np.ndarray,
    which_ds: str,
    *,
    base_dir: str | Path = ".",
) -> np.ndarray:
    """Полный перенос логики DNscan.m.

    which_ds:
    - DSxoz
    - DSyoz
    - DSalpha
    - DS3d
    """
    table = _ensure_scan_table(data_manager, base_dir=base_dir)
    mode = which_ds.lower()

    if mode == "dsxoz":
        return _scan_2d_cut(data_manager, table, np.asarray(xx, dtype=float), phi_value=0.0)
    if mode == "dsyoz":
        return _scan_2d_cut(data_manager, table, np.asarray(xx, dtype=float), phi_value=90.0)
    if mode == "dsalpha":
        alpha = float(data_manager.state.pattern.alpha)
        return _scan_2d_cut(data_manager, table, np.asarray(xx, dtype=float), phi_value=alpha)
    if mode == "ds3d":
        return _scan_3d(table, np.asarray(xx, dtype=float), np.asarray(yy, dtype=float))

    raise ValueError(f"Неизвестный тип which_ds='{which_ds}'")


# ---------------------------- DNscan internals ----------------------------
def _scan_2d_cut(data_manager: DataManager, table: np.ndarray, xx: np.ndarray, *, phi_value: float) -> np.ndarray:
    """Логика веток DSxoz/DSyoz/DSalpha из DNscan.m."""
    orig_shape = xx.shape
    xx = np.round(np.ravel(xx), 2)
    if xx.size < 2:
        step_th = 0.1
    else:
        step_th = abs(xx[1] - xx[0])

    row_mask = np.isclose(table[:, 0], phi_value, atol=1e-12)
    if not np.any(row_mask):
        raise ValueError(f"В таблице ДС отсутствует сечение для phi={phi_value}")

    ds_r_db = table[row_mask, 2]
    ds_l_db = ds_r_db[1:][::-1]
    ds_s_db = np.concatenate([ds_l_db, ds_r_db])

    # MATLAB: DS_int=interp1(DSs_dB,(1:0.1:rtw),'spline')
    grid_src = np.arange(1.0, ds_s_db.size + 1.0, 1.0)
    grid_dense = np.arange(1.0, ds_s_db.size + 1e-12, 0.1)
    ds_int = np.interp(grid_dense, grid_src, ds_s_db)

    gr = np.arange(-89.0, 89.0 + 1e-12, 0.01)
    n = min(gr.size, ds_int.size)
    gr = gr[:n]
    ds_int = ds_int[:n]

    # Логика выбора значений с клиппингом по границам.
    out = np.interp(xx, gr, ds_int, left=ds_int[0], right=ds_int[-1])

    # Сохраняем как вектор той же формы, что и xx на входе DNscan.
    return out.reshape(orig_shape)


def _scan_3d(table: np.ndarray, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    """Логика ветки DS3d из DNscan.m (включая перестановку осей yy,xx)."""
    # Построение базовых матриц (как в MATLAB):
    # f = repmat(0:90,891,1)
    # t = repmat((0:0.1:89)',1,91)
    f = np.tile(np.arange(0.0, 91.0, 1.0), (891, 1))
    t = np.tile(np.arange(0.0, 89.0 + 1e-12, 0.1)[:, None], (1, 91))
    x = np.rad2deg(np.arcsin(np.sin(np.deg2rad(t)) * np.sin(np.deg2rad(f))))
    y = np.rad2deg(np.arcsin(np.sin(np.deg2rad(t)) * np.cos(np.deg2rad(f))))

    # Перенос цикла сборки ds(s,i)=importDS(j,3) при i=flip(1:91)
    col3 = table[:, 2]
    expected = 891 * 91
    if col3.size < expected:
        raise ValueError(f"Недостаточно строк для DS3d: {col3.size}, требуется минимум {expected}")

    ds = np.zeros((891, 91), dtype=float)
    k = 0
    for i in range(90, -1, -1):
        ds[:, i] = col3[k : k + 891]
        k += 891

    bds = np.vstack((np.flipud(ds), ds))
    cds = np.hstack((np.fliplr(bds), bds))
    bxds = np.vstack((np.flipud(x), x))
    cxds = np.hstack((-np.fliplr(bxds), bxds))
    byds = np.vstack((-np.flipud(y), y))
    cyds = np.hstack((-np.fliplr(byds), byds))

    # MATLAB: DS_dB = griddata(CXds,CYds,Cds,yy,xx)
    # Важная деталь: yy,xx местами намеренно (комментарий исходника).
    points = np.column_stack((cxds.ravel(), cyds.ravel()))
    values = cds.ravel()
    query_points = np.column_stack((yy.ravel(), xx.ravel()))

    if _scipy_griddata is not None:
        out = _scipy_griddata(points, values, query_points, method="linear")
        # Вне выпуклой оболочки fallback на nearest
        nan_mask = np.isnan(out)
        if np.any(nan_mask):
            out[nan_mask] = _nearest_values(points, values, query_points[nan_mask])
    else:
        out = _nearest_values(points, values, query_points)

    return out.reshape(xx.shape)


def _nearest_values(points: np.ndarray, values: np.ndarray, query_points: np.ndarray) -> np.ndarray:
    """Fallback интерполяции для DS3d без SciPy: nearest-neighbor."""
    out = np.empty(query_points.shape[0], dtype=float)
    chunk = 4096
    for i in range(0, query_points.shape[0], chunk):
        q = query_points[i : i + chunk]
        dist2 = (points[:, 0][None, :] - q[:, 0][:, None]) ** 2 + (points[:, 1][None, :] - q[:, 1][:, None]) ** 2
        out[i : i + chunk] = values[np.argmin(dist2, axis=1)]
    return out


def _ensure_scan_table(data_manager: DataManager, *, base_dir: str | Path) -> np.ndarray:
    table = data_manager.get_calc_array("scan_pattern_raw")
    if table is None:
        return import_scan_pattern(data_manager, base_dir=base_dir)
    return np.asarray(table, dtype=float)


def _resolve_scan_pattern_type(*, requested: ScanPatternType, liter: int) -> ScanPatternType:
    if requested != ScanPatternType.AUTO:
        return requested
    if liter in _MIDDLE_LIT:
        return ScanPatternType.MIDDLE
    if liter in _LOWER_LIT:
        return ScanPatternType.LOWER
    if liter in _UPPER_LIT:
        return ScanPatternType.UPPER
    raise ValueError(
        f"Для литеры {liter} не определён тип диаграммы сканирования в логике DNscan.m. "
        "Установите pattern.scan_pattern_type вручную (lower/middle/upper)."
    )


def _resolve_scan_file_path(*, base_dir: str | Path, requested_path: str, scan_type: ScanPatternType) -> Path:
    if requested_path:
        return Path(requested_path)
    return Path(base_dir) / _DEFAULT_FILE_BY_TYPE[scan_type]


def _load_scan_table(file_path: Path) -> np.ndarray:
    if not file_path.exists():
        raise FileNotFoundError(f"Файл диаграммы сканирования не найден: {file_path}")

    if file_path.suffix.lower() == ".npy":
        table = np.load(file_path)
    else:
        delimiter = "," if file_path.suffix.lower() == ".csv" else None
        table = np.loadtxt(file_path, delimiter=delimiter)

    table = np.asarray(table, dtype=float)
    if table.ndim != 2 or table.shape[1] < 3:
        raise ValueError(f"Неверный формат диаграммы сканирования {file_path}: ожидалась таблица Nx3+, получено {table.shape}")
    return table


__all__ = ["import_scan_pattern", "calculate_scan_pattern_db"]
