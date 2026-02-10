from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

from DataManager import DataManager, ScanPatternType

_MIDDLE_LIT = {39, 41, 51, 53, 63}
_LOWER_LIT = {1, 27, 29}
_UPPER_LIT = {65, 75, 77, 101}

_DEFAULT_FILE_BY_TYPE: Dict[ScanPatternType, str] = {
    ScanPatternType.MIDDLE: "DS_s.txt",
    ScanPatternType.LOWER: "DS_n.txt",
    ScanPatternType.UPPER: "DS_v.txt",
}


def import_scan_pattern(data_manager: DataManager, *, base_dir: str | Path = ".") -> np.ndarray:
    """Импортировать диаграмму сканирования по логике DNscan.m.

    Выбор типа ДС сохранён:
    - AUTO -> определяется по литере (middle/lower/upper);
    - LOWER/MIDDLE/UPPER -> принудительный выбор.

    После импорта обновляет DataManager:
    - state.pattern.scan_pattern_loaded
    - state.pattern.scan_pattern_file_path
    - calc_arrays.scan_pattern_raw
    - results.scan_pattern
    """

    state = data_manager.state
    pattern = state.pattern
    arr = state.array

    scan_type = _resolve_scan_pattern_type(
        requested=pattern.scan_pattern_type,
        liter=int(arr.liter),
    )

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

    file_name = _DEFAULT_FILE_BY_TYPE[scan_type]
    return Path(base_dir) / file_name


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
        raise ValueError(
            f"Неверный формат диаграммы сканирования {file_path}: ожидалась таблица Nx3+, получено {table.shape}"
        )

    return table


__all__ = ["import_scan_pattern"]
