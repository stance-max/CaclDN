"Централизованное хранение данных"

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import asdict, dataclass, field, asdict
from enum import Enum
import threading
import json
import pickle
from datetime import datetime


class GridType(Enum):
    "Типы сетки расположения облучаталей"
    RECTANGULAR = "Прямогоульгая"
    HEXAGONAL = "Гексагональная"

@dataclass
class ArrayParameters(Enum):
    "Параметры антенной решетки"
    # Геометрические параметры
    grid_type: GridType = GridType.HEXAGONAL    # Выбор типа расположения сетки 
    nxe: int = 256          # Кол-во эл-тов по Х
    nye: int = 128          # Кол-во эл-тов по Y
    dx: float = 17.7        # Шаг решетки по X, мм
    dy: float = 20.4        # Шаг решетки по Y, мм
    nxb: int = 3            # Кол-во опорных балок по Х
    nyb: int = 0            # Кол-во опорных балок по Y
    xb: float = 37.0        # Ширина опорных балок по Х, мм
    yb: float = 37.0        # Ширина опорных балок по Y, мм
    # Частотные параматры
    liter: int = 41         # Номер литеры
    c_light: float = 3e8    # Скорость света, м/с
    freq: float = (6.5+(liter-1)*2.01)*1e9  # Частота, Гц
    wavelenght: float = (c_light/freq)      # Длина волны, м

@dataclass
class PatterParameters:
    "Параметры диаграммы направленности" 
    # Углы отклонения луча, град
    th0: float = 0.0
    ph0: float = 0.0
    # Диапазоны отображения, град
    th_min: float = -89.0
    th_max: float = 89.0
    ph_min: float = 0.0
    ph_max: float = 360.0
    # Шаги построения
    step_2d: float = 0.1
    step_3d: float = 0.1
    alpha: float = 0.0 # Сечение, град

@dataclass
class AFRParameters:
    "Параметры амлитудно-фазового распределения"





