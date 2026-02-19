from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from dataclasses import asdict
from typing import Dict

from DataManager import (
    ARType,
    DataManager,
    GridType,
    PRType,
    ScanPatternType,
)
from CalcAFR import calculate_afr
from CalcDN2 import calculate_dn2_sections
from CalcDN3 import calculate_dn3
from CalcPCH import calculate_pch
from CalcParam import calculate_parameters
from plot import render_all


class CalcDnGUI:
    """GUI в стиле CW_new.m (структура вкладок и панелей)."""

    def __init__(self, data_manager: DataManager) -> None:
        self.dm = data_manager
        self.root = tk.Tk()
        self.root.title("Калькулятор ДН АФАР")
        self.root.geometry("1200x800")
        self.root.resizable(False, False)

        self.vars: Dict[str, tk.Variable] = {}

        self._build_menu()
        self._build_tabs()
        self._build_params_tab()

    def run(self) -> None:
        self.root.mainloop()

    # -------------------------- build --------------------------
    def _build_menu(self) -> None:
        m = tk.Menu(self.root)
        m.add_command(label="Расчет", command=self._on_start)
        m.add_command(label="Очистить", command=self._on_clear)
        m.add_command(label="Сохранить", command=self._on_save)
        self.root.config(menu=m)

    def _build_tabs(self) -> None:
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True)

        self.tabs = {}
        titles = [
            ("param", "Параметры"),
            ("ar", "АР"),
            ("fr", "ФР"),
            ("xoz", "Сечение XOZ"),
            ("yoz", "Сечение YOZ"),
            ("alpha", "Сечение Альфа"),
            ("pch", "ПХ"),
            ("s3d", "Суммарная 3D"),
            ("ra3d", "Азимутальная 3D"),
            ("re3d", "Угломестная 3D"),
            ("grid", "Сетка излуч-й"),
        ]
        for key, title in titles:
            f = ttk.Frame(self.notebook)
            self.notebook.add(f, text=title)
            self.tabs[key] = f

    def _build_params_tab(self) -> None:
        tab = self.tabs["param"]

        self._build_array_panel(tab)
        self._build_pattern_panel(tab)
        self._build_afr_panel(tab)
        self._build_flags_panel(tab)

    def _build_array_panel(self, parent: ttk.Frame) -> None:
        st = self.dm.state.array
        lf = ttk.LabelFrame(parent, text="Параметры решетки")
        lf.place(x=10, y=10, width=560, height=220)

        self._add_combo(lf, "Тип сетки:", "array.grid_type", [g.value for g in GridType], st.grid_type.value, 5, 5, 60, 22, 110, 22, 2)
        self._add_entry(lf, "Литер:", "array.liter", st.liter, 190, 5, 40, 22, 100, 22, 2)
        self._add_entry(lf, "Каналы OX, шт:", "array.nxe", st.nxe, 5, 45)
        self._add_entry(lf, "Каналы OY, шт", "array.nye", st.nye, 190, 45)
        self._add_entry(lf, "Шаг OX, м:", "array.dx", st.dx, 5, 80)
        self._add_entry(lf, "Шаг OY, м:", "array.dy", st.dy, 190, 80)
        self._add_entry(lf, "Балки OX, шт:", "array.nxb", st.nxb, 5, 115)
        self._add_entry(lf, "Балки OX, шт:", "array.nyb", st.nyb, 190, 115)
        self._add_entry(lf, "Шир.балк OX, м:", "array.xb", st.xb, 5, 150)
        self._add_entry(lf, "Шир.балк OY, м:", "array.yb", st.yb, 190, 150,)
        self._add_check(lf, "xkord imported", "afr.xkord_imported", st.xkord_imported, 10, 175)
        self._add_check(lf, "ykord imported", "afr.ykord_imported", st.ykord_imported, 190, 175)

    def _build_pattern_panel(self, parent: ttk.Frame) -> None:
        st = self.dm.state.pattern
        lf = ttk.LabelFrame(parent, text="Параметры ДН")
        lf.place(x=580, y=10, width=610, height=220)

        self._add_entry(lf, "Th0", "pattern.th0", st.th0, 10, 10)
        self._add_entry(lf, "Ph0", "pattern.ph0", st.ph0, 170, 10)
        self._add_entry(lf, "ThMin", "pattern.th_min", st.th_min, 10, 45)
        self._add_entry(lf, "ThMax", "pattern.th_max", st.th_max, 170, 45)
        self._add_entry(lf, "Step2D", "pattern.step_2d", st.step_2d, 10, 80)
        self._add_entry(lf, "Step3D", "pattern.step_3d", st.step_3d, 170, 80)
        self._add_entry(lf, "Alpha", "pattern.alpha", st.alpha, 330, 80)

        self._add_check(lf, "Use scan pattern (2D/3D)", "pattern.use_scanpattern", st.use_scanpattern, 10, 120)
        self._add_combo(
            lf,
            "Тип ДС",
            "pattern.scan_pattern_type",
            [x.value for x in ScanPatternType],
            st.scan_pattern_type.value,
            10,
            150,
        )

    def _build_afr_panel(self, parent: ttk.Frame) -> None:
        st = self.dm.state.afr
        lf = ttk.LabelFrame(parent, text="Параметры АФР")
        lf.place(x=10, y=240, width=560, height=240)

        self._add_combo(lf, "AR", "afr.amplitude_type", [x.value for x in ARType], st.amplitude_type.value, 10, 10)
        self._add_combo(lf, "PR", "afr.phase_type", [x.value for x in PRType], st.phase_type.value, 10, 45)
        self._add_entry(lf, "AR dop", "afr.ar_dop", st.ar_dop, 10, 80)
        self._add_entry(lf, "FR dop1", "afr.fr_dop1", st.fr_dop1, 10, 115)
        self._add_entry(lf, "FR dop2", "afr.fr_dop2", st.fr_dop2, 190, 115)

        self._add_check(lf, "mask imported", "afr.mask_imported", st.mask_imported, 10, 150)
        #self._add_check(lf, "xkord imported", "afr.xkord_imported", st.xkord_imported, 10, 175)
        #self._add_check(lf, "ykord imported", "afr.ykord_imported", st.ykord_imported, 190, 175)

    def _build_flags_panel(self, parent: ttk.Frame) -> None:
        st = self.dm.state.plots
        lf = ttk.LabelFrame(parent, text="Настройки расчета")
        lf.place(x=580, y=240, width=610, height=240)

        self._add_check(lf, "2D ДН", "plots.open_2d", st.open_2d, 10, 10)
        self._add_check(lf, "ПХ", "plots.open_pch", st.open_pch, 10, 35)
        self._add_check(lf, "Пеленг", "plots.open_peling", st.open_peling, 10, 60)
        self._add_check(lf, "3D ДН", "plots.open_3d", st.open_3d, 10, 90)
        self._add_check(lf, "3D сум", "plots.open_3d_sum", st.open_3d_sum, 170, 90)
        self._add_check(lf, "3D азимут", "plots.open_3d_rz_az", st.open_3d_rz_az, 300, 90)
        self._add_check(lf, "3D угломест", "plots.open_3d_rz_el", st.open_3d_rz_el, 460, 90)

        ttk.Button(lf, text="Применить", command=self._apply_state).place(x=10, y=180, width=120)

    # ---------------------- controls helpers ----------------------
    def _place_labeled_control(
        self,
        parent: ttk.Frame,
        label: str,
        x: int,
        y: int,
        *,
        label_width: int,
        label_height: int,
        spacing: int,
    ) -> tuple[ttk.Label, int]:
        lbl = ttk.Label(parent, text=label)
        lbl.place(x=x, y=y, width=label_width, height=label_height)
        control_x = x + label_width + spacing
        return lbl, control_x

    def _add_entry(
        self,
        parent: ttk.Frame,
        label: str,
        key: str,
        value: object,
        x: int,
        y: int,
        label_width: int = 80,
        label_height: int = 22,
        control_width: int = 120,
        control_height: int = 22,
        spacing: int = 0,
    ) -> None:
        _, control_x = self._place_labeled_control(
            parent,
            label,
            x,
            y,
            label_width=label_width,
            label_height=label_height,
            spacing=spacing,
        )
        v = tk.StringVar(value=str(value))
        self.vars[key] = v
        ttk.Entry(parent, textvariable=v).place(x=control_x, y=y, width=control_width, height=control_height)

    def _add_combo(
        self,
        parent: ttk.Frame,
        label: str,
        key: str,
        values: list[str],
        value: str,
        x: int,
        y: int,
        label_width: int = 80,
        label_height: int = 22,
        control_width: int = 140,
        control_height: int = 22,
        spacing: int = 0,
    ) -> None:
        _, control_x = self._place_labeled_control(
            parent,
            label,
            x,
            y,
            label_width=label_width,
            label_height=label_height,
            spacing=spacing,
        )
        v = tk.StringVar(value=value)
        self.vars[key] = v
        ttk.Combobox(parent, textvariable=v, values=values, state="readonly").place(
            x=control_x,
            y=y,
            width=control_width,
            height=control_height,
        )

    def _add_check(
        self,
        parent: ttk.Frame,
        label: str,
        key: str,
        value: bool,
        x: int,
        y: int,
        label_width: int = 160,
        label_height: int = 22,
        control_width: int = 22,
        control_height: int = 22,
        spacing: int = 0,
    ) -> None:
        lbl, control_x = self._place_labeled_control(
            parent,
            label,
            x,
            y,
            label_width=label_width,
            label_height=label_height,
            spacing=spacing,
        )
        v = tk.BooleanVar(value=bool(value))
        self.vars[key] = v
        check = ttk.Checkbutton(parent, variable=v)
        check.place(x=control_x, y=y, width=control_width, height=control_height)
        lbl.bind("<Button-1>", lambda _event, var=v: var.set(not var.get()))

    def _add_text(
        self,
        parent: ttk.Frame,
        text: str,
        x: int,
        y: int,
        width: int = 120,
        height: int = 22,
    ) -> None:
        ttk.Label(parent, text=text).place(x=x, y=y, width=width, height=height)

    # ---------------------- actions ----------------------
    def _apply_state(self) -> None:
        array_patch = {
            "grid_type": GridType(self.vars["array.grid_type"].get()),
            "nxe": int(float(self.vars["array.nxe"].get())),
            "nye": int(float(self.vars["array.nye"].get())),
            "dx": float(self.vars["array.dx"].get()),
            "dy": float(self.vars["array.dy"].get()),
            "nxb": int(float(self.vars["array.nxb"].get())),
            "nyb": int(float(self.vars["array.nyb"].get())),
            "xb": float(self.vars["array.xb"].get()),
            "yb": float(self.vars["array.yb"].get()),
            "liter": int(float(self.vars["array.liter"].get())),
        }
        pattern_patch = {
            "th0": float(self.vars["pattern.th0"].get()),
            "ph0": float(self.vars["pattern.ph0"].get()),
            "th_min": float(self.vars["pattern.th_min"].get()),
            "th_max": float(self.vars["pattern.th_max"].get()),
            "step_2d": float(self.vars["pattern.step_2d"].get()),
            "step_3d": float(self.vars["pattern.step_3d"].get()),
            "alpha": float(self.vars["pattern.alpha"].get()),
            "use_scanpattern": bool(self.vars["pattern.use_scanpattern"].get()),
            "scan_pattern_type": ScanPatternType(self.vars["pattern.scan_pattern_type"].get()),
        }
        afr_patch = {
            "amplitude_type": ARType(self.vars["afr.amplitude_type"].get()),
            "phase_type": PRType(self.vars["afr.phase_type"].get()),
            "ar_dop": float(self.vars["afr.ar_dop"].get()),
            "fr_dop1": float(self.vars["afr.fr_dop1"].get()),
            "fr_dop2": float(self.vars["afr.fr_dop2"].get()),
            "mask_imported": bool(self.vars["afr.mask_imported"].get()),
            "xkord_imported": bool(self.vars["afr.xkord_imported"].get()),
            "ykord_imported": bool(self.vars["afr.ykord_imported"].get()),
        }
        plots_patch = {
            "open_2d": bool(self.vars["plots.open_2d"].get()),
            "open_pch": bool(self.vars["plots.open_pch"].get()),
            "open_peling": bool(self.vars["plots.open_peling"].get()),
            "open_3d": bool(self.vars["plots.open_3d"].get()),
            "open_3d_sum": bool(self.vars["plots.open_3d_sum"].get()),
            "open_3d_rz_az": bool(self.vars["plots.open_3d_rz_az"].get()),
            "open_3d_rz_el": bool(self.vars["plots.open_3d_rz_el"].get()),
        }

        self.dm.patch_state("array", array_patch)
        self.dm.patch_state("pattern", pattern_patch)
        self.dm.patch_state("afr", afr_patch)
        self.dm.patch_state("plots", plots_patch)

        self.dm.set("gui.last_apply", {
            "array": asdict(self.dm.state.array),
            "pattern": asdict(self.dm.state.pattern),
            "afr": asdict(self.dm.state.afr),
            "plots": asdict(self.dm.state.plots),
        })

    def _on_start(self) -> None:
        self._apply_state()
        self.dm.set("gui.command", "start")

        try:
            calculate_afr(self.dm)
            calculate_dn2_sections(self.dm, calc_sum=True, calc_diff=True)

            if self.dm.state.plots.open_3d:
                calculate_dn3(self.dm)

            calculate_pch(self.dm)
            calculate_parameters(self.dm)
            render_all(self.dm, self.tabs)
        except Exception as exc:
            messagebox.showerror("Ошибка расчёта", str(exc), parent=self.root)
            self.dm.set("gui.last_error", str(exc))
            return

        messagebox.showinfo("CalcDN", "Расчёт завершён и графики обновлены.", parent=self.root)

    def _on_save(self) -> None:
        self.dm.set("gui.command", "save")

    def _on_clear(self) -> None:
        self.dm.set("gui.command", "clear")


def build_gui(data_manager: DataManager) -> CalcDnGUI:
    return CalcDnGUI(data_manager)