from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from DataManager import ARType, DataManager, MaskType, PRType
from CalcCord import calculate_coordinate_grid


PHASE_QUANT_STEP_DEG = 5.625


try:
    from scipy.signal.windows import chebwin as _chebwin
except Exception:  # pragma: no cover - optional dependency
    _chebwin = None


def calculate_afr(data_manager: DataManager, *, random_seed: int | None = None) -> Dict[str, Any]:
    """Вычислить АФР и обновить DataManager.

    Перенос логики из AFR.m:
    - амплитудное распределение;
    - маска включения;
    - фазовое распределение;
    - квантование фазы по шагу 5.625°;
    - фазовые матрицы Rx/Ry.

    Единицы измерения: метры (dx/dy/xb/yb/Xkord/Ykord/R).
    """

    state = data_manager.state
    arr = state.array
    afr = state.afr
    pat = state.pattern

    nx, ny = int(arr.nxe), int(arr.nye)
    dx, dy = float(arr.dx), float(arr.dy)
    xb, yb = float(arr.xb), float(arr.yb)
    nxb, nyb = int(arr.nxb), int(arr.nyb)

    xkord, ykord = _resolve_coordinates(data_manager, nx=nx, ny=ny, dx=dx, dy=dy)
    lx = nx * dx
    ly = ny * dy
    b = nxb * xb

    amplitude_raw = _build_amplitude(
        afr_type=afr.amplitude_type,
        ar_dop=float(afr.ar_dop),
        nx=nx,
        ny=ny,
        dx=dx,
        dy=dy,
        b=b,
        xkord=xkord,
        ykord=ykord,
        file_path=afr.amplitude_file_path,
    )

    mask = _resolve_mask(data_manager, nx=nx, ny=ny)
    amplitude = mask * amplitude_raw

    phase_dist_deg = _build_phase_distribution_deg(
        phase_type=afr.phase_type,
        fr_dop1=float(afr.fr_dop1),
        fr_dop2=float(afr.fr_dop2),
        nx=nx,
        ny=ny,
        dx=dx,
        dy=dy,
        lx=lx,
        ly=ly,
        xkord=xkord,
        ykord=ykord,
        wavelength_m=float(arr.wavelength_m),
        random_seed=random_seed,
        file_path=afr.phase_file_path,
    )

    phase_offset_rad = np.zeros((ny, nx), dtype=float)
    if pat.th0 != 0.0 or pat.ph0 != 0.0:
        phase_offset_rad = -arr.wave_k * (
            xkord * np.sin(np.deg2rad(pat.th0)) + ykord * np.sin(np.deg2rad(pat.ph0))
        )

    phase_deg = phase_dist_deg + np.rad2deg(phase_offset_rad)
    ph_code = np.rint(phase_deg / PHASE_QUANT_STEP_DEG)
    phase_quant_deg = np.mod(ph_code * PHASE_QUANT_STEP_DEG, 360.0)
    phase_rad = np.deg2rad(ph_code * PHASE_QUANT_STEP_DEG)

    phase_rx_rad = phase_rad.copy()
    phase_ry_rad = phase_rad.copy()
    phase_rx_rad[:, nx // 2 :] += np.pi
    phase_ry_rad[ny // 2 :, :] += np.pi

    phase_rx_deg = np.rad2deg(phase_rx_rad)
    phase_ry_deg = np.rad2deg(phase_ry_rad)

    # Обновление ключевых массивов в DataManager.
    data_manager.set_calc_array("xkord", xkord)
    data_manager.set_calc_array("ykord", ykord)
    data_manager.set_calc_array("mask", mask)
    data_manager.set_calc_array("amplitude", amplitude)
    data_manager.set_calc_array("phase_rad", phase_rad)
    data_manager.set_calc_array("phase_deg", phase_quant_deg)
    data_manager.set_calc_array("phase_rx_rad", phase_rx_rad)
    data_manager.set_calc_array("phase_ry_rad", phase_ry_rad)
    data_manager.set_calc_array("phase_rx_deg", phase_rx_deg)
    data_manager.set_calc_array("phase_ry_deg", phase_ry_deg)

    result = {
        "phase_quant_step_deg": PHASE_QUANT_STEP_DEG,
        "amplitude_type": afr.amplitude_type.value,
        "phase_type": afr.phase_type.value,
        "array_m": {"nx": nx, "ny": ny, "dx": dx, "dy": dy, "xb": xb, "yb": yb, "nxb": nxb, "nyb": nyb},
    }
    data_manager.set_calc_param("afr_phase_quant_step_deg", float(PHASE_QUANT_STEP_DEG))
    data_manager.set_calc_param("afr_nx", int(nx))
    data_manager.set_calc_param("afr_ny", int(ny))

    data_manager.set_result("afr", result)
    data_manager.set("afr_result", result)

    return {
        "amplitude": amplitude,
        "mask": mask,
        "phase_rad": phase_rad,
        "phase_deg": phase_quant_deg,
        "phase_rx_rad": phase_rx_rad,
        "phase_ry_rad": phase_ry_rad,
        "phase_rx_deg": phase_rx_deg,
        "phase_ry_deg": phase_ry_deg,
        "meta": result,
    }


def _resolve_coordinates(data_manager: DataManager, *, nx: int, ny: int, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    xkord, ykord = calculate_coordinate_grid(data_manager)
    x_arr = np.asarray(xkord, dtype=float)
    y_arr = np.asarray(ykord, dtype=float)
    if x_arr.shape != (ny, nx) or y_arr.shape != (ny, nx):
        raise ValueError(f"Неверный размер координатной сетки: x={x_arr.shape}, y={y_arr.shape}, ожидается {(ny, nx)}")
    return x_arr, y_arr


def _build_amplitude(
    *,
    afr_type: ARType,
    ar_dop: float,
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    b: float,
    xkord: np.ndarray,
    ykord: np.ndarray,
    file_path: str,
) -> np.ndarray:
    if afr_type == ARType.UNIFORM:
        return np.full((ny, nx), ar_dop if ar_dop != 0 else 1.0, dtype=float)

    if afr_type == ARType.COS_ON_PEDESTAL:
        pedestal = ar_dop
        return pedestal + (1.0 - pedestal) * np.cos(xkord * np.pi / ((nx - 1) * dx + b)) * np.cos(ykord * np.pi / ((ny - 0.5) * dy))

    if afr_type == ARType.DOLPH_CHEBYSHEV:
        if _chebwin is None:
            raise RuntimeError("Для Dolph-Chebyshev требуется scipy (scipy.signal.windows.chebwin).")
        wy = _chebwin(ny, at=max(ar_dop, 1.0))
        wx = _chebwin(nx, at=max(ar_dop, 1.0))
        return np.outer(wy, wx)

    if afr_type == ARType.HANN:
        wy = np.hanning(ny)
        wx = np.hanning(nx)
        return np.outer(wy, wx)

    if afr_type == ARType.HAMMING:
        wy = np.hamming(ny)
        wx = np.hamming(nx)
        return np.outer(wy, wx)

    if afr_type == ARType.BLACKMAN:
        a = ar_dop
        l = np.arange(ny, 0, -1, dtype=float)
        k = np.arange(nx, 0, -1, dtype=float)
        wy = (1 - a) / 2 - 0.5 * np.cos(2 * np.pi * l / ny) + 0.5 * a * np.cos(4 * np.pi * l / ny)
        wx = (1 - a) / 2 - 0.5 * np.cos(2 * np.pi * k / nx) + 0.5 * a * np.cos(4 * np.pi * k / nx)
        return np.outer(wy, wx)

    if afr_type == ARType.KAISER:
        beta = 0.0
        if ar_dop > 50:
            beta = 0.1102 * (ar_dop - 8.7)
        elif ar_dop >= 21:
            beta = 0.5842 * (ar_dop - 21) ** 0.4 + 0.07886 * (ar_dop - 21)
        return np.outer(np.kaiser(ny, beta), np.kaiser(nx, beta))

    if afr_type == ARType.FROM_FILE:
        return _load_array_from_file(file_path=file_path, expected_shape=(ny, nx), name="amplitude")

    raise ValueError(f"Неизвестный тип амплитудного распределения: {afr_type}")


def _build_phase_distribution_deg(
    *,
    phase_type: PRType,
    fr_dop1: float,
    fr_dop2: float,
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    lx: float,
    ly: float,
    xkord: np.ndarray,
    ykord: np.ndarray,
    wavelength_m: float,
    random_seed: int | None,
    file_path: str,
) -> np.ndarray:
    rng = np.random.default_rng(random_seed)

    if phase_type == PRType.UNIFORM:
        return np.zeros((ny, nx), dtype=float)

    if phase_type == PRType.CIRCULAR:
        r = fr_dop1
        phase = _rand_symmetric_phase(ny=ny, nx=nx, rng=rng)
        inside = np.hypot(xkord, ykord) < r
        phase[inside] = 0.0
        return phase

    if phase_type == PRType.PYRAMIDAL:
        return np.abs(xkord / dx) * fr_dop1 + np.abs(ykord / dy) * fr_dop2

    if phase_type == PRType.SPHERICAL:
        r = fr_dop1
        phase = _rand_symmetric_phase(ny=ny, nx=nx, rng=rng)
        rho2 = xkord**2 + ykord**2
        inside = rho2 < r**2
        phase[inside] = 360.0 * (r - np.sqrt(np.maximum(r**2 - rho2[inside], 0.0))) / wavelength_m
        return phase

    if phase_type == PRType.QUASI_FIBONACCI:
        hnx = nx // 2
        hny = ny // 2
        a = fr_dop1 * np.cumsum(np.arange(hnx, dtype=float))
        b = fr_dop2 * np.cumsum(np.arange(hny, dtype=float))
        quarter = b[:, None] + a[None, :]
        right_half = np.vstack((np.flipud(quarter), quarter))
        return np.hstack((np.fliplr(right_half), right_half))

    if phase_type == PRType.QUADRATIC:
        return ((xkord / lx) ** 2 * fr_dop1 + (ykord / ly) ** 2 * fr_dop2) * 360.0

    if phase_type == PRType.PRIME_NUMBERS:
        hnx = nx // 2
        hny = ny // 2
        primes = _primes(1000)
        if len(primes) < max(hnx, hny):
            raise ValueError("Недостаточно простых чисел для выбранной размерности решетки.")
        str1 = primes[:hnx] * fr_dop1
        col1 = primes[:hny] * fr_dop2
        sector1 = col1[:, None] + str1[None, :]
        sector2 = np.hstack((np.fliplr(sector1), sector1))
        return np.vstack((np.flipud(sector2), sector2))

    if phase_type == PRType.FROM_FILE:
        return _load_array_from_file(file_path=file_path, expected_shape=(ny, nx), name="phase")

    raise ValueError(f"Неизвестный тип фазового распределения: {phase_type}")


def _resolve_mask(data_manager: DataManager, *, nx: int, ny: int) -> np.ndarray:
    afr = data_manager.state.afr

    if afr.mask_type == MaskType.FULL:
        return np.ones((ny, nx), dtype=float)

    if afr.mask_type == MaskType.IMPORTED:
        if not afr.mask_imported:
            raise ValueError("Выбран тип маски IMPORTED, но флаг mask_imported=False.")
        mask = data_manager.get_calc_array("mask")
        if mask is None:
            raise ValueError("Флаг mask_imported=True, но маска отсутствует в DataManager.calc_arrays.mask.")
        mask_arr = np.asarray(mask, dtype=float)
        if mask_arr.shape != (ny, nx):
            raise ValueError(f"Неверный размер импортированной маски: {mask_arr.shape}, ожидается {(ny, nx)}")
        return mask_arr

    if afr.mask_type == MaskType.ROCKET:
        mask = np.zeros((ny, nx), dtype=float)
        # 64 центральных БП (перенос индексации gn(2,64), gn(3,64) из AFR.m).
        if nx >= 192:
            mask[:, 64:192] = 1.0
        else:
            left = max(0, nx // 2 - nx // 4)
            right = min(nx, nx // 2 + nx // 4)
            mask[:, left:right] = 1.0
        return mask

    raise ValueError(f"Неизвестный тип mask_type={afr.mask_type}")


def _load_array_from_file(*, file_path: str, expected_shape: tuple[int, int], name: str) -> np.ndarray:
    if not file_path:
        raise ValueError(f"Пустой путь к файлу для {name}.")

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл {name} не найден: {file_path}")

    ext = path.suffix.lower()
    if ext == ".npy":
        arr = np.load(path)
    else:
        delimiter = "," if ext in {".csv"} else None
        arr = np.loadtxt(path, delimiter=delimiter)

    arr = np.asarray(arr, dtype=float)
    if arr.shape != expected_shape:
        raise ValueError(f"Неверный размер {name}: {arr.shape}, ожидается {expected_shape}")
    return arr


def _rand_symmetric_phase(*, ny: int, nx: int, rng: np.random.Generator) -> np.ndarray:
    qy, qx = ny // 2, nx // 2
    ff = 180.0 * rng.choice(np.array([-1.0, 1.0], dtype=float), size=(qy, qx))
    top = np.vstack((ff, np.flipud(ff)))
    return np.hstack((np.fliplr(top), top))


def _primes(limit: int) -> np.ndarray:
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[:2] = False
    for p in range(2, int(limit**0.5) + 1):
        if sieve[p]:
            sieve[p * p : limit + 1 : p] = False
    return np.flatnonzero(sieve).astype(float)


__all__ = ["calculate_afr", "PHASE_QUANT_STEP_DEG"]
