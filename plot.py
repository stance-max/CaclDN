from __future__ import annotations

from typing import Any

from DataManager import DataManager


def plot_stub(data_manager: DataManager, target: str, payload: Any | None = None) -> None:
    """Заглушка построения графиков.

    Полноценная реализация будет добавлена позднее.
    """
    data_manager.set("plot.last_target", target)
    if payload is not None:
        data_manager.set("plot.last_payload", payload)
