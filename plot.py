from __future__ import annotations

import tkinter as tk
from typing import Any, Mapping

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from DataManager import DataManager


def render_all(data_manager: DataManager, tabs: Mapping[str, tk.Widget]) -> None:
    """Построить доступные графики в GUI-вкладках с учетом флагов."""
    st = data_manager.state
    ca = st.calc_arrays
    plots = st.plots
    afr = st.afr

    _plot_ar(tabs.get("ar"), np.asarray(ca.amplitude) if ca.amplitude is not None else None, show=(plots.open_ar or afr.show_ar_plot))
    _plot_fr(tabs.get("fr"), np.asarray(ca.phase_deg) if ca.phase_deg is not None else None, show=(plots.open_fr or afr.show_fr_plot))

    angles = np.asarray(ca.dn2_angles_deg) if ca.dn2_angles_deg is not None else None
    _plot_cut(
        tabs.get("xoz"),
        angles,
        _arr(ca.dn2_sum_xoz_db),
        _arr(ca.dn2_diff_xoz_db),
        title="Наклонная плоскость XOZ",
        show=plots.open_2d,
    )
    _plot_cut(
        tabs.get("yoz"),
        angles,
        _arr(ca.dn2_sum_yoz_db),
        _arr(ca.dn2_diff_yoz_db),
        title="Вертикальная плоскость YOZ",
        show=plots.open_2d,
    )

    _plot_alpha(tabs.get("alpha"))
    _plot_pch(tabs.get("pch"), _arr(ca.pch_fi_n_deg), _arr(ca.pch_nakl), _arr(ca.pch_fi_v_deg), _arr(ca.pch_vert), show=plots.open_pch)

    _plot_3d(tabs.get("s3d"), _arr(ca.dn3_x_deg), _arr(ca.dn3_y_deg), _arr(ca.dn3_sum_db), "Суммарная 3D", show=plots.open_3d and plots.open_3d_sum)
    _plot_3d(tabs.get("ra3d"), _arr(ca.dn3_x_deg), _arr(ca.dn3_y_deg), _arr(ca.dn3_rz_az_db), "Азимутальная 3D", show=plots.open_3d and plots.open_3d_rz_az)
    _plot_3d(tabs.get("re3d"), _arr(ca.dn3_x_deg), _arr(ca.dn3_y_deg), _arr(ca.dn3_rz_el_db), "Угломестная 3D", show=plots.open_3d and plots.open_3d_rz_el)

    _plot_grid(tabs.get("grid"), _arr(ca.xkord), _arr(ca.ykord), show=plots.open_grid)


def _arr(v: Any) -> np.ndarray | None:
    return None if v is None else np.asarray(v)


def _clear_tab(tab: tk.Widget | None) -> None:
    if tab is None:
        return
    for w in tab.winfo_children():
        w.destroy()


def _draw_text(tab: tk.Widget | None, text: str) -> None:
    if tab is None:
        return
    _clear_tab(tab)
    lbl = tk.Label(tab, text=text, anchor="center")
    lbl.pack(fill="both", expand=True)


def _draw_figure(tab: tk.Widget | None, fig: Figure) -> None:
    if tab is None:
        return
    _clear_tab(tab)
    canvas = FigureCanvasTkAgg(fig, master=tab)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)


def _plot_ar(tab: tk.Widget | None, amplitude: np.ndarray | None, *, show: bool) -> None:
    if not show:
        _draw_text(tab, "График АР выключен флагами.")
        return
    if amplitude is None:
        _draw_text(tab, "Нет данных АР.")
        return
    fig = Figure(figsize=(8, 5), dpi=100)
    ax = fig.add_subplot(111)
    im = ax.imshow(amplitude, aspect="auto", origin="lower", cmap="viridis")
    ax.set_title("Амплитудное распределение")
    fig.colorbar(im, ax=ax)
    _draw_figure(tab, fig)


def _plot_fr(tab: tk.Widget | None, phase_deg: np.ndarray | None, *, show: bool) -> None:
    if not show:
        _draw_text(tab, "График ФР выключен флагами.")
        return
    if phase_deg is None:
        _draw_text(tab, "Нет данных ФР.")
        return
    fig = Figure(figsize=(8, 5), dpi=100)
    ax = fig.add_subplot(111)
    im = ax.imshow(phase_deg, aspect="auto", origin="lower", cmap="twilight")
    ax.set_title("Фазовое распределение, град")
    fig.colorbar(im, ax=ax)
    _draw_figure(tab, fig)


def _plot_cut(tab: tk.Widget | None, angles: np.ndarray | None, sum_db: np.ndarray | None, diff_db: np.ndarray | None, *, title: str, show: bool) -> None:
    if not show:
        _draw_text(tab, "2D сечения выключены флагами.")
        return
    if angles is None or sum_db is None:
        _draw_text(tab, "Нет данных 2D сечений.")
        return
    fig = Figure(figsize=(8, 5), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot(angles, sum_db, "r", label="Сум")
    if diff_db is not None:
        ax.plot(angles, diff_db, "g--", label="Раз")
    ax.grid(True, which="both", alpha=0.4)
    ax.set_title(title)
    ax.set_xlabel("Угол, град")
    ax.set_ylabel("Уровень, дБ")
    ax.set_xlim(-90, 90)
    ax.set_ylim(-60, 1)
    ax.legend(loc="best")
    _draw_figure(tab, fig)


def _plot_alpha(tab: tk.Widget | None) -> None:
    _draw_text(tab, "Сечение Альфа будет добавлено в plot.py позже.")


def _plot_pch(tab: tk.Widget | None, fi_n: np.ndarray | None, phx: np.ndarray | None, fi_v: np.ndarray | None, phy: np.ndarray | None, *, show: bool) -> None:
    if not show:
        _draw_text(tab, "ПХ выключена флагами.")
        return
    if fi_n is None or phx is None or fi_v is None or phy is None:
        _draw_text(tab, "Нет данных ПХ.")
        return
    fig = Figure(figsize=(9, 4), dpi=100)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.plot(fi_n, phx, "b")
    ax1.grid(True, alpha=0.4)
    ax1.set_title("Наклонная ПХ")
    ax1.set_xlabel("Град")
    ax1.set_ylabel("Отн. ед.")
    ax2.plot(fi_v, phy, "r")
    ax2.grid(True, alpha=0.4)
    ax2.set_title("Вертикальная ПХ")
    ax2.set_xlabel("Град")
    ax2.set_ylabel("Отн. ед.")
    fig.tight_layout()
    _draw_figure(tab, fig)


def _plot_3d(tab: tk.Widget | None, x_deg: np.ndarray | None, y_deg: np.ndarray | None, z_db: np.ndarray | None, title: str, *, show: bool) -> None:
    if not show:
        _draw_text(tab, f"{title}: выключено флагами.")
        return
    if x_deg is None or y_deg is None or z_db is None:
        _draw_text(tab, f"{title}: нет данных.")
        return
    fig = Figure(figsize=(8, 5), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(x_deg, y_deg, z_db, cmap="viridis", linewidth=0, antialiased=True)
    ax.set_title(title)
    ax.set_xlabel("X, град")
    ax.set_ylabel("Y, град")
    ax.set_zlabel("дБ")
    fig.colorbar(surf, ax=ax, shrink=0.6)
    _draw_figure(tab, fig)


def _plot_grid(tab: tk.Widget | None, xkord: np.ndarray | None, ykord: np.ndarray | None, *, show: bool) -> None:
    if not show:
        _draw_text(tab, "Сетка излучателей выключена флагами.")
        return
    if xkord is None or ykord is None:
        _draw_text(tab, "Нет данных координатной сетки.")
        return
    fig = Figure(figsize=(8, 5), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot(xkord, ykord, ".r", markersize=2)
    ax.grid(True, alpha=0.4)
    ax.set_title("Координатная сетка излучателей")
    ax.set_xlabel("X, м")
    ax.set_ylabel("Y, м")
    ax.axis("equal")
    _draw_figure(tab, fig)
