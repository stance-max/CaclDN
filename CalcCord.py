from __future__ import annotations

from typing import Tuple

import numpy as np

from DataManager import DataManager, GridType


def calculate_coordinate_grid(data_manager: DataManager) -> Tuple[np.ndarray, np.ndarray]:
    """Рассчитать или взять из импорта координатные сетки Xkord/Ykord.

    Логика:
    - если включён импорт xkord/ykord, берём массивы только из DataManager.calc_arrays;
    - иначе считаем по мотивам KordSetkaOX.m и KordSetkaOY.m (в метрах);
    - результат всегда сохраняется обратно в DataManager.
    """

    state = data_manager.state
    arr = state.array
    afr = state.afr

    nx, ny = int(arr.nxe), int(arr.nye)
    dx, dy = float(arr.dx), float(arr.dy)
    nxb, nyb = int(arr.nxb), int(arr.nyb)
    xb, yb = float(arr.xb), float(arr.yb)

    imported_x = bool(afr.xkord_imported)
    imported_y = bool(afr.ykord_imported)

    x_raw = data_manager.get_calc_array("xkord")
    y_raw = data_manager.get_calc_array("ykord")

    if imported_x or imported_y:
        if not (imported_x and imported_y):
            raise ValueError("Для импортированных координат должны быть установлены оба флага: xkord_imported и ykord_imported.")
        if x_raw is None or y_raw is None:
            raise ValueError("Флаги импорта координат установлены, но xkord/ykord отсутствуют в DataManager.calc_arrays.")

        xkord = np.asarray(x_raw, dtype=float)
        ykord = np.asarray(y_raw, dtype=float)
        _validate_shapes(xkord, ykord, ny=ny, nx=nx)
    else:
        xkord = _build_xkord(nx=nx, ny=ny, dx=dx, nxbalk=nxb, xb=xb)
        ykord = _build_ykord(
            grid_type=arr.grid_type,
            nx=nx,
            ny=ny,
            dy=dy,
            nybalk=nyb,
            yb=yb,
        )

    data_manager.set_calc_array("xkord", xkord)
    data_manager.set_calc_array("ykord", ykord)
    data_manager.set_calc_param("coordinates_imported", bool(imported_x and imported_y))
    data_manager.set_calc_param("coordinates_nx", int(nx))
    data_manager.set_calc_param("coordinates_ny", int(ny))
    data_manager.set_result(
        "coordinates",
        {
            "source": "imported" if imported_x and imported_y else "computed",
            "shape": [ny, nx],
            "units": "m",
            "grid_type": arr.grid_type.value if isinstance(arr.grid_type, GridType) else str(arr.grid_type),
        },
    )

    return xkord, ykord


def _build_xkord(*, nx: int, ny: int, dx: float, nxbalk: int, xb: float) -> np.ndarray:
    if nx % 2 != 0:
        raise ValueError("KordSetkaOX поддерживает только чётное Nx.")

    half = nx // 2
    n = np.arange(1, nx + 1)

    if nxbalk == 0 or xb == 0:
        x_line = np.where(
            n <= half,
            -((half - n) * dx + dx / 2.0),
            (n - half - 1) * dx + dx / 2.0,
        )
        return np.tile(x_line, (ny, 1))

    nsec = nxbalk + 1
    nxsec = nx / nsec
    if int(nxsec) != nxsec:
        raise ValueError("Некорректное соотношение балок и элементов по оси OX.")
    nxsec = int(nxsec)

    left_n = np.arange(1, half + 1)
    right_n = np.arange(half + 1, nx + 1)

    if nxbalk % 2 == 1:
        left_k = half - left_n
        left_kb = np.array([np.count_nonzero(np.mod(np.arange(v, half + 1), nxsec) == 0) - 1 for v in left_n])
        left_x = -xb / 2.0 - dx / 2.0 - left_k * dx - left_kb * xb

        right_k = right_n - (half + 1)
        right_kb = np.array([np.count_nonzero(np.mod(np.arange(half + 1, v + 1), nxsec) == 1) - 1 for v in right_n])
        right_x = xb / 2.0 + dx / 2.0 + right_k * dx + right_kb * xb
    else:
        left_k = half - left_n
        left_kb = np.array([np.count_nonzero(np.mod(np.arange(v, half + 1), nxsec) == 0) for v in left_n])
        left_x = -dx / 2.0 - left_k * dx - left_kb * xb

        right_k = right_n - (half + 1)
        right_kb = np.array([np.count_nonzero(np.mod(np.arange(half + 1, v + 1), nxsec) == 1) for v in right_n])
        right_x = dx / 2.0 + right_k * dx + right_kb * xb

    x_line = np.concatenate([left_x, right_x])
    return np.tile(x_line, (ny, 1))


def _build_ykord(*, grid_type: GridType | str, nx: int, ny: int, dy: float, nybalk: int, yb: float) -> np.ndarray:
    if ny % 2 != 0:
        raise ValueError("KordSetkaOY поддерживает только чётное Ny.")

    is_hex = (grid_type == GridType.HEXAGONAL) or (str(grid_type) == GridType.HEXAGONAL.value)
    c = 0.0 if is_hex else 1.0
    g = 1.0 if is_hex else 0.0

    half = ny // 2
    ykord = np.zeros((ny, nx), dtype=float)

    if nybalk == 0 or yb == 0:
        for n in range(1, nx + 1):
            top_a = 1.0 if (is_hex and (n % 2 == 1)) else 0.0
            bot_a = 1.0 if (is_hex and (n % 2 == 0)) else 0.0

            top_k = np.arange(half - 1, -1, -1, dtype=float)
            bot_k = np.arange(0, half, dtype=float)

            ykord[:half, n - 1] = -top_a * dy / 2.0 - top_k * dy - g * dy / 4.0 - c * dy / 2.0
            ykord[half:, n - 1] = bot_a * dy / 2.0 + bot_k * dy + g * dy / 4.0 + c * dy / 2.0
        return ykord

    nsec = nybalk + 1
    nysec = ny / nsec
    if int(nysec) != nysec:
        raise ValueError("Некорректное соотношение балок и элементов по оси OY.")
    nysec = int(nysec)

    top_i = np.arange(1, half + 1)
    bot_i = np.arange(half + 1, ny + 1)

    for n in range(1, nx + 1):
        top_a = 1.0 if (is_hex and (n % 2 == 1)) else 0.0
        bot_a = 1.0 if (is_hex and (n % 2 == 0)) else 0.0

        top_k = (half - top_i).astype(float)
        bot_k = (bot_i - (half + 1)).astype(float)

        if nybalk % 2 == 1:
            top_kb = np.array([np.count_nonzero(np.mod(np.arange(v, half + 1), nysec) == 0) - 1 for v in top_i], dtype=float)
            bot_kb = np.array([np.count_nonzero(np.mod(np.arange(half + 1, v + 1), nysec) == 1) - 1 for v in bot_i], dtype=float)

            y_top = -yb / 2.0 - top_kb * yb - top_a * dy / 2.0 - top_k * dy - g * dy / 4.0 - c * dy / 2.0
            y_bot = yb / 2.0 + bot_kb * yb + bot_a * dy / 2.0 + bot_k * dy + g * dy / 4.0 + c * dy / 2.0
        else:
            top_kb = np.array([np.count_nonzero(np.mod(np.arange(v, half + 1), nysec) == 0) for v in top_i], dtype=float)
            bot_kb = np.array([np.count_nonzero(np.mod(np.arange(half + 1, v + 1), nysec) == 1) for v in bot_i], dtype=float)

            y_top = -top_kb * yb - top_a * dy / 2.0 - top_k * dy - g * dy / 4.0 - c * dy / 2.0
            y_bot = bot_kb * yb + bot_a * dy / 2.0 + bot_k * dy + g * dy / 4.0 + c * dy / 2.0

        ykord[:half, n - 1] = y_top
        ykord[half:, n - 1] = y_bot

    return ykord


def _validate_shapes(xkord: np.ndarray, ykord: np.ndarray, *, ny: int, nx: int) -> None:
    expected = (ny, nx)
    if xkord.shape != expected or ykord.shape != expected:
        raise ValueError(f"Неверный размер координат: x={xkord.shape}, y={ykord.shape}, ожидается {expected}")


__all__ = ["calculate_coordinate_grid"]
