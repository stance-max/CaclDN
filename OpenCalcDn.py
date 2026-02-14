from __future__ import annotations

from DataManager import DataManager
from gui import build_gui


def open_calc_dn(data_manager: DataManager | None = None) -> DataManager:
    """Точка входа открытия интерфейса и управления состоянием."""
    dm = data_manager or DataManager()
    app = build_gui(dm)
    app.run()
    return dm


if __name__ == "__main__":
    open_calc_dn()
