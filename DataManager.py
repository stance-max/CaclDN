from __future__ import annotations

"""Централизованное хранение данных.

Содержание:
- Тип геометрии расположения излучателей - GridType
- Параметры антенной решетки - ArrayParameters
- Параметры диаграммы направленности - PatternParameters
- Тип амплитудного распределения - ARType
- Тип фазового распределения - PRType
- Параметры амлитудно-фазового распределения - AFRParameters
- Ключевые массивы расчета и сохранения результатов - CalculationArrays
- Флаги визуализации и построения графиков - PlotFlags
- Полное типизированное состояние приложения - AppState
- Потокобезопасный менеджер данных - DataManager
"""

import math
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
import json
import pickle
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, MutableMapping, Optional


class GridType(str, Enum):
    "Тип геометрии расположения излучателей"

    RECTANGULAR = "rectangular"  # Прямоугольная сетка
    HEXAGONAL = "hexagonal"      # Гексагональная сетка


class ScanPatternType(str, Enum):
    """Тип импортируемой диаграммы сканирования (DNscan.m)."""

    AUTO = "auto"      # Выбор по литере
    LOWER = "lower"    # Нижняя лита (DS_n)
    MIDDLE = "middle"  # Средняя лита (DS_s)
    UPPER = "upper"    # Верхняя лита (DS_v)


@dataclass
class ArrayParameters:
    "Параметры антенной решетки"
        # ---- Геометрические параметры ----
    grid_type: GridType = GridType.HEXAGONAL    # Выбор типа расположения сетки 
    nxe: int = 256          # Кол-во эл-тов по Х
    nye: int = 128          # Кол-во эл-тов по Y
    dx: float = 0.0177      # Шаг решетки по X, м
    dy: float = 0.0204      # Шаг решетки по Y, м
    nxb: int = 3            # Кол-во опорных балок по Х
    nyb: int = 0            # Кол-во опорных балок по Y
    xb: float = 0.037       # Ширина опорных балок по Х, м
    yb: float = 0.037       # Ширина опорных балок по Y, м
        # ---- Частотные параматры ----
    liter: int = 41                            # Номер литеры
    c_light: float = 3e8                   # Скорость света (м/с)

    # Частота/длина волны как переменные
    freq_hz: float = field(default=0.0)        # Частота (Гц)
    wavelength_m: float = field(default=0.0)   # Длина волны (м)
    wavelength_mm: float = field(default=0.0)  # Длина волны (мм)
    wave_k: float = field(default=0.0)         # Волновое число k=2π/λ (рад/м)

    def __post_init__(self) -> None:
        # Формулы перенесены из MATLAB CalcDN.m:
        # freq = (6.5 + (lit-1)*2.01) * 10^9
        # lamda_m = c_light/freq  [в м]
        self.freq_hz = (6.5 + (self.liter - 1) * 2.01) * 1e9
        self.wavelength_m = self.c_light / self.freq_hz
        self.wavelength_mm = self.wavelength_m * 1000.0
        self.wave_k = 2.0 * math.pi / self.wavelength_m


@dataclass
class PatternParameters:
    "Параметры диаграммы направленности" 
        # ---- Углы отклонения луча, град ----
    th0: float = 0.0
    ph0: float = 0.0
        # ---- Диапазоны отображения, град ----
    th_min: float = -89.0
    th_max: float = 89.0
    ph_min: float = 0.0
    ph_max: float = 360.0
        # ---- Шаги построения ---- 
    step_2d: float = 0.1
    step_3d: float = 0.1
    alpha: float = 0.0 # Сечение, град

    # ---- Импорт диаграммы сканирования (DNscan.m) ----
    scan_pattern_type: ScanPatternType = ScanPatternType.AUTO
    scan_pattern_loaded: bool = False
    scan_pattern_file_path: str = ""

    # Флаг режима расчёта 2D-сечений:
    # False -> только множитель решётки (без ДС),
    # True  -> расчёт с учётом диаграммы сканирования элемента (ДС).
    use_scanpattern: bool = False


class ARType(str, Enum):
    "Тип амплитудного распределения (АР) в соответствии с ChoseAR.m."

    UNIFORM = "uniform"                  # 1) Равномерное
    COS_ON_PEDESTAL = "cos_on_pedestal"  # 2) cos на пьедестале
    DOLPH_CHEBYSHEV = "dolph_chebyshev"  # 3) Дольф-Чебышев
    HANN = "hann"                        # 4) Окно Ханна
    HAMMING = "hamming"                  # 5) Окно Хэмминга
    BLACKMAN = "blackman"                # 6) Окно Блэкмана
    KAISER = "kaiser"                    # 7) Окно Кайзера
    FROM_FILE = "from_file"              # 8) Из файла


class PRType(str, Enum):
    """Тип фазового распределения (ФР) в соответствии с ChoseFR.m."""

    UNIFORM = "uniform"                # 1) Равномерное
    CIRCULAR = "circular"              # 2) Круговое
    PYRAMIDAL = "pyramidal"            # 3) Пирамидальное
    SPHERICAL = "spherical"            # 4) Сферическое
    QUASI_FIBONACCI = "quasi_fibonacci"  # 5) Квазифибоначчи
    QUADRATIC = "quadratic"            # 6) Квадратическое
    PRIME_NUMBERS = "prime_numbers"    # 7) Простые числа
    FROM_FILE = "from_file"            # 8) Из файла


class MaskType(str, Enum):
    """Тип маски включения излучателей (AFR.m -> ChMask)."""

    FULL = "full"            # 1) Все полотно
    IMPORTED = "imported"    # 2) Импортированная маска
    ROCKET = "rocket"        # 3) 64 центральных БП (ракетный луч)


@dataclass
class AFRParameters:
    "Параметры амлитудно-фазового распределения"
        # ---- Общие флаги АФР ----
    enabled: bool = True                  # Аналог btn.SetAFR: применять АФР при расчёте
    show_ar_plot: bool = False            # Аналог btn.AFR.AR: показывать график АР
    show_fr_plot: bool = False            # Аналог btn.AFR.FR: показывать график ФР
    use_db_scale: bool = False            # Параметр вида АР (0/1 или дБ), если задействован
    
          # ---- Дополнительные служебные флаги ----
    # Используются для явной фиксации того, что данные реально загружены из файла.
    amplitude_loaded_from_file: bool = False
    phase_loaded_from_file: bool = False

    # ---- Типы распределений из AFR.m/ChoseAR.m/ChoseFR.m ----
    amplitude_type: ARType = ARType.UNIFORM
    phase_type: PRType = PRType.UNIFORM
    mask_type: MaskType = MaskType.FULL

    # ---- Флаги и пути импортируемых массивов (обновляются импортёром) ----
    mask_imported: bool = False
    xkord_imported: bool = False
    ykord_imported: bool = False
    mask_file_path: str = ""
    xkord_file_path: str = ""
    ykord_file_path: str = ""
    
        # ---- Универсальные "доп" параметры GUI ----
    # ARdop из ChoseAR.m: один параметр, смысл зависит от amplitude_type:
    # - COS_ON_PEDESTAL: высота подставки (0..1)
    # - DOLPH_CHEBYSHEV: подавление БЛ (10..100 дБ)
    # - BLACKMAN: коэффициент a (обычно 0.16)
    # - KAISER: подавление БЛ (10..100 дБ)
    ar_dop: float = 0.0

    # FRdop1/FRdop2 из ChoseFR.m: смысл зависит от phase_type:
    # - CIRCULAR, SPHERICAL: fr_dop1 = R (м)
    # - PYRAMIDAL, QUASI_FIBONACCI, QUADRATIC, PRIME_NUMBERS:
    #       fr_dop1 = ddx (град), fr_dop2 = ddy (град)
    fr_dop1: float = 0.0
    fr_dop2: float = 0.0
    
        # ---- Файловые источники (case 8 для AR/FR) ----
    amplitude_file_path: str = ""         # Путь к файлу амплитудного распределения
    phase_file_path: str = ""             # Путь к файлу фазового распределения

@dataclass(slots=True)
class CalculationArrays:
    """Ключевые массивы расчёта и сохранения результатов.

    Все поля допускают numpy-массивы/списки; тип оставлен `Any`,
    чтобы не ограничивать формат хранения на этапе миграции.
    """

    # Координаты решётки
    xkord: Any = None          # Xkord
    ykord: Any = None          # Ykord

    # Амплитудные данные
    amplitude: Any = None      # Am
    mask: Any = None           # Mask

    # Фазовые распределения в радианах
    phase_rad: Any = None      # phase (rad)
    phase_rx_rad: Any = None   # phaseRx (rad)
    phase_ry_rad: Any = None   # phaseRy (rad)

    # Фазовые распределения в градусах
    phase_deg: Any = None      # phase (deg)
    phase_rx_deg: Any = None   # phaseRx (deg)
    phase_ry_deg: Any = None   # phaseRy (deg)

    # Импортированная диаграмма сканирования (DNscan)
    scan_pattern_raw: Any = None  # таблица Nx3: [phi(or alpha), theta, dB]

    # 2D-сечения ДН (CalcDN2) — ключевые массивы
    dn2_angles_deg: Any = None
    dn2_sum_xoz_linear: Any = None
    dn2_sum_yoz_linear: Any = None
    dn2_sum_xoz_db: Any = None
    dn2_sum_yoz_db: Any = None
    dn2_diff_xoz_linear: Any = None
    dn2_diff_yoz_linear: Any = None
    dn2_diff_xoz_db: Any = None
    dn2_diff_yoz_db: Any = None

    # 3D ДН (CalcDN3) — ключевые массивы
    dn3_x_deg: Any = None
    dn3_y_deg: Any = None
    dn3_sum_db: Any = None
    dn3_rz_az_db: Any = None
    dn3_rz_el_db: Any = None
    
@dataclass(slots=True)
class PlotFlags:
    "Флаги визуализации и построения графиков"
    open_2d: bool = True      # 2D сечения XOZ/YOZ
    open_peleng: bool = False # Пеленгационное направление
    open_3d: bool = False     # 3D расчёты разрешены
    open_3d_sum: bool = True  # Суммарная 3D
    open_3d_rz_az: bool = False  # Азимутальная 3D
    open_3d_rz_el: bool = False  # Угломестная 3D
    open_section: bool = False  # Произвольное сечение
    open_grid: bool = False   # Координатная сетка излучателей
    open_ar: bool = False     # График амплитудного распределения
    open_fr: bool = False     # График фазового распределения
    open_sko: bool = True     # СКО/доп. метрики (если используются)

    
@dataclass(slots=True)
class AppState:
    """Полное типизированное состояние приложения.

    Секции:
    - array: параметры антенной решётки;
    - pattern: параметры расчёта ДН;
    - afr: параметры АФР;
    - plots: флаги отображения;
    - calc_arrays: ключевые массивы расчёта/сохранения;
    - results: результаты вычислений (аналог MATLAB Res.*);
    - runtime: временные данные рабочего сеанса.
    """

    array: ArrayParameters = field(default_factory=ArrayParameters)
    pattern: PatternParameters = field(default_factory=PatternParameters)
    afr: AFRParameters = field(default_factory=AFRParameters)
    plots: PlotFlags = field(default_factory=PlotFlags)
    calc_arrays: CalculationArrays = field(default_factory=CalculationArrays)
    results: Dict[str, Any] = field(default_factory=dict)
    runtime: Dict[str, Any] = field(default_factory=dict)


# Колбек подписки: key изменения, новое value, ссылка на DataManager.
Subscriber = Callable[[str, Any, "DataManager"], None]


class DataManager:
    """Потокобезопасный менеджер данных для расчёта и будущего GUI."""

    def __init__(self, initial_state: Optional[AppState] = None) -> None:
        # Основное состояние (типизированное, для бизнес-логики).
        self._state: AppState = initial_state or AppState()
        # Свободное key-value хранилище (временные/служебные переменные).
        self._store: Dict[str, Any] = {}
        # Таблица подписчиков: ключ -> список обработчиков.
        self._subscribers: Dict[str, List[Subscriber]] = {}
        # RLock для потокобезопасности в GUI/рабочих потоках.
        self._lock = threading.RLock()

    # ---------------------- Базовые операции со store ---------------------
    def set(self, key: str, value: Any, *, notify: bool = True) -> None:
        """Сохранить значение по ключу верхнего уровня."""
        with self._lock:
            self._store[key] = value
        if notify:
            self._emit(key, value)

    def get(self, key: str, default: Any = None) -> Any:
        """Получить значение по ключу верхнего уровня."""
        with self._lock:
            return self._store.get(key, default)

    def update(self, values: MutableMapping[str, Any], *, notify: bool = True) -> None:
        """Пакетно обновить несколько ключей в store."""
        with self._lock:
            for key, value in values.items():
                self._store[key] = value
        if notify:
            for key, value in values.items():
                self._emit(key, value)

    def remove(self, key: str, *, notify: bool = True) -> Any:
        """Удалить ключ и вернуть его старое значение (или None)."""
        with self._lock:
            prev = self._store.pop(key, None)
        if notify:
            self._emit(key, None)
        return prev

    def clear_store(self, *, notify: bool = False) -> None:
        """Очистить runtime-store полностью."""
        with self._lock:
            self._store.clear()
        if notify:
            self._emit("store", {})

    # ----------------------- Типизированный AppState ----------------------
    @property
    def state(self) -> AppState:
        """Текущее типизированное состояние приложения."""
        with self._lock:
            return self._state

    def replace_state(self, state: AppState, *, notify: bool = True) -> None:
        """Полностью заменить AppState новым объектом."""
        with self._lock:
            self._state = state
        if notify:
            self._emit("state", self._state)

    def patch_state(self, section: str, values: MutableMapping[str, Any], *, notify: bool = True) -> None:
        """Обновить часть состояния (`array`, `pattern`, `afr`, `plots`)."""
        with self._lock:
            section_obj = getattr(self._state, section)
            for field_name, field_value in values.items():
                setattr(section_obj, field_name, field_value)

            # После изменения параметров решётки пересчитываем зависимые поля
            # частоты/длины волны как обычные переменные.
            if section == "array":
                section_obj.freq_hz = (6.5 + (section_obj.liter - 1) * 2.01) * 1e9
                section_obj.wavelength_m = section_obj.c_light / section_obj.freq_hz
                section_obj.wavelength_mm = section_obj.wavelength_m * 1000.0
                section_obj.wave_k = 2.0 * math.pi / section_obj.wavelength_m

        if notify:
            self._emit(f"state.{section}", section_obj)

    def set_result(self, name: str, value: Any, *, notify: bool = True) -> None:
        """Сохранить вычисленный результат (аналог MATLAB `Res.<name>`)."""
        with self._lock:
            self._state.results[name] = value
        if notify:
            self._emit(f"results.{name}", value)

    def get_result(self, name: str, default: Any = None) -> Any:
        """Получить результат по имени."""
        with self._lock:
            return self._state.results.get(name, default)

    def set_calc_array(self, name: str, value: Any, *, notify: bool = True) -> None:
        """Сохранить ключевой расчётный массив по имени поля `calc_arrays`."""
        with self._lock:
            if not hasattr(self._state.calc_arrays, name):
                raise AttributeError(f"Unknown calc array field: {name}")
            setattr(self._state.calc_arrays, name, value)
        if notify:
            self._emit(f"calc_arrays.{name}", value)

    def get_calc_array(self, name: str, default: Any = None) -> Any:
        """Получить ключевой массив по имени поля `calc_arrays`."""
        with self._lock:
            if not hasattr(self._state.calc_arrays, name):
                return default
            value = getattr(self._state.calc_arrays, name)
            return default if value is None else value

    # ----------------------------- Подписки -------------------------------
    def subscribe(self, key: str, callback: Subscriber) -> None:
        """Подписка на обновления.

        Примеры ключей:
        - `state.array`
        - `results.afr`
        - `*` (все изменения)
        """
        with self._lock:
            self._subscribers.setdefault(key, []).append(callback)

    def unsubscribe(self, key: str, callback: Subscriber) -> None:
        """Удалить подписку callback для заданного ключа."""
        with self._lock:
            callbacks = self._subscribers.get(key, [])
            self._subscribers[key] = [cb for cb in callbacks if cb is not callback]

    def _emit(self, key: str, value: Any) -> None:
        """Разослать уведомления подписчикам точного ключа и wildcard `*`."""
        with self._lock:
            callbacks = list(self._subscribers.get(key, []))
            wildcard = list(self._subscribers.get("*", []))

        for callback in callbacks + wildcard:
            callback(key, value, self)

    # ---------------------------- Снимки/IO -------------------------------
    def snapshot(self) -> Dict[str, Any]:
        """Сформировать сериализуемый снимок текущего состояния."""
        with self._lock:
            state_payload = _to_serializable(self._state)
            store_payload = _to_serializable(self._store)

        return {
            "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
            "state": state_payload,
            "store": store_payload,
        }

    def load_snapshot(self, payload: Dict[str, Any], *, notify: bool = True) -> None:
        """Восстановить состояние из словаря снимка."""
        state_raw = payload.get("state", {})
        restored_state = AppState(
            array=_array_from_dict(state_raw.get("array", {})),
            pattern=_pattern_from_dict(state_raw.get("pattern", {})),
            afr=_afr_from_dict(state_raw.get("afr", {})),
            plots=PlotFlags(**state_raw.get("plots", {})),
            calc_arrays=CalculationArrays(**state_raw.get("calc_arrays", {})),
            results=state_raw.get("results", {}),
            runtime=state_raw.get("runtime", {}),
        )

        with self._lock:
            self._state = restored_state
            self._store = payload.get("store", {})

        if notify:
            self._emit("state", self._state)
            self._emit("store", self._store)

    def dump_json(self, file_path: str | Path) -> None:
        """Сохранить снимок в JSON-файл."""
        path = Path(file_path)
        path.write_text(json.dumps(self.snapshot(), ensure_ascii=False, indent=2), encoding="utf-8")

    def load_json(self, file_path: str | Path, *, notify: bool = True) -> None:
        """Загрузить снимок из JSON-файла."""
        path = Path(file_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.load_snapshot(payload, notify=notify)

    def dump_pickle(self, file_path: str | Path) -> None:
        """Сохранить снимок в бинарный pickle-файл."""
        path = Path(file_path)
        path.write_bytes(pickle.dumps(self.snapshot()))

    def load_pickle(self, file_path: str | Path, *, notify: bool = True) -> None:
        """Загрузить снимок из бинарного pickle-файла."""
        path = Path(file_path)
        payload = pickle.loads(path.read_bytes())
        self.load_snapshot(payload, notify=notify)


def _to_serializable(value: Any) -> Any:
    """Рекурсивно преобразовать объекты в JSON-friendly структуру."""
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {k: _to_serializable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_serializable(v) for v in value]
    return value


def _pattern_from_dict(payload: Dict[str, Any]) -> PatternParameters:
    """Восстановить PatternParameters с enum-типом ScanPatternType."""
    normalized = dict(payload)
    if "scan_pattern_type" in normalized and not isinstance(normalized["scan_pattern_type"], ScanPatternType):
        normalized["scan_pattern_type"] = ScanPatternType(normalized["scan_pattern_type"])
    # Backward compatibility: старое имя флага -> новое.
    if "dn2_use_scan_pattern" in normalized and "use_scanpattern" not in normalized:
        normalized["use_scanpattern"] = normalized.pop("dn2_use_scan_pattern")
    return PatternParameters(**normalized)


def _array_from_dict(payload: Dict[str, Any]) -> ArrayParameters:
    """Восстановить ArrayParameters, корректно обработав GridType."""
    normalized = dict(payload)
    if "grid_type" in normalized and not isinstance(normalized["grid_type"], GridType):
        normalized["grid_type"] = GridType(normalized["grid_type"])
    return ArrayParameters(**normalized)


def _afr_from_dict(payload: Dict[str, Any]) -> AFRParameters:
    """Восстановить AFRParameters с enum-типами АР/ФР."""
    normalized = dict(payload)
    if "amplitude_type" in normalized and not isinstance(normalized["amplitude_type"], ARType):
        normalized["amplitude_type"] = ARType(normalized["amplitude_type"])
    if "phase_type" in normalized and not isinstance(normalized["phase_type"], PRType):
        normalized["phase_type"] = PRType(normalized["phase_type"])
    if "mask_type" in normalized and not isinstance(normalized["mask_type"], MaskType):
        normalized["mask_type"] = MaskType(normalized["mask_type"])
    return AFRParameters(**normalized)
