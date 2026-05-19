import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from tqdm import tqdm
import tkinter as tk
from tkinter import ttk, messagebox


# ============================================================
# LOCAL ANALYTICAL SOLUTION
# ============================================================

def velocity_local(t, v0, tau0, p):
    return (
        _a * (np.sin(tau0 + t) - np.sin(tau0))
        - p * t
        + v0
    )


def position_local(t, x0, v0, tau0, p):
    return (
        -_a * (np.cos(tau0 + t) - np.cos(tau0))
        + v0 * t
        - 0.5 * p * t**2
        - _a * t * np.sin(tau0)
        + x0
    )


def gap_function(t, x0, v0, tau0, p):
    tau = tau0 + t
    x = position_local(t, x0, v0, tau0, p)
    return x + _mu * np.cos(tau)


# ============================================================
# IMPACT SEARCH
# ============================================================

def find_impact_time(x0, v0, tau0, p, t_max=100.0, dt=0.02):
    t1 = 1e-10
    g1 = gap_function(t1, x0, v0, tau0, p)
    t = t1 + dt

    while t <= t_max:
        g2 = gap_function(t, x0, v0, tau0, p)
        if g1 * g2 <= 0:
            try:
                t_impact = brentq(
                    lambda x: gap_function(x, x0, v0, tau0, p),
                    t - dt, t,
                    xtol=1e-12, rtol=1e-12
                )
                return t_impact
            except ValueError:
                pass
        g1 = g2
        t += dt

    return None


# ============================================================
# SINGLE IMPACT STEP
# ============================================================

def do_impact_step(x0, v0, tau0, p):
    t_impact = find_impact_time(x0, v0, tau0, p)
    if t_impact is None:
        return None

    tau_impact = tau0 + t_impact
    x_minus = position_local(t_impact, x0, v0, tau0, p)
    v_minus = velocity_local(t_impact, v0, tau0, p)
    v_plus = -_R * v_minus + (1 + _R) * _mu * np.sin(tau_impact)

    return tau_impact, x_minus, v_plus


# ============================================================
# MAIN SIMULATION
# ============================================================

def run_simulation(params):
    global _R, _mu, _lam, _a

    _R   = params['R']
    _mu  = params['mu']
    _lam = params['lam']
    _a   = _mu * _lam

    p_start     = params['p_start']
    p_end       = params['p_end']
    p_step      = params['p_step']
    N_transient = params['N_transient']
    N_keep      = params['N_keep']

    ic_tau0    = params['ic_tau0']
    ic_v0      = params['ic_v0']
    ic_x0_auto = params['ic_x0_auto']
    ic_x0      = params['ic_x0']

    p_vals = []
    v_vals = []

    p_values = np.arange(p_start, p_end, p_step)

    for p in tqdm(p_values, desc="Bifurcation"):
        tau0 = ic_tau0
        x0   = -_mu * np.cos(tau0) if ic_x0_auto else ic_x0
        v0   = ic_v0

        # Transient
        n_done = 0
        while n_done < N_transient:
            result = do_impact_step(x0, v0, tau0, p)
            if result is None:
                break
            tau0, x0, v0 = result
            n_done += 1

        # Record
        velocities = []
        while len(velocities) < N_keep:
            result = do_impact_step(x0, v0, tau0, p)
            if result is None:
                break
            tau0, x0, v0 = result
            velocities.append(v0)

        p_vals.extend([p] * len(velocities))
        v_vals.extend(velocities)

    plt.figure(figsize=(12, 7))
    plt.scatter(p_vals, v_vals, s=0.2, alpha=0.6)
    plt.xlabel('p', fontsize=14)
    plt.ylabel('Velocity after impact', fontsize=14)
    plt.title('Bifurcation diagram', fontsize=15)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ============================================================
# GUI
# ============================================================

def build_gui():
    root = tk.Tk()
    root.title("Bifurcation Diagram — Parameters")
    root.resizable(False, False)

    BG       = "#1a1a2e"
    FG       = "#e0e0e0"
    ACCENT   = "#4fc3f7"
    ENTRY_BG = "#162032"
    FONT_L   = ("Courier New", 10)
    FONT_E   = ("Courier New", 11)

    root.configure(bg=BG)

    # ── Header ───────────────────────────────────────────────
    tk.Frame(root, bg=ACCENT, height=4).pack(fill="x")

    title_frame = tk.Frame(root, bg=BG, pady=14)
    title_frame.pack(fill="x", padx=24)
    tk.Label(
        title_frame, text="BIFURCATION DIAGRAM",
        font=("Courier New", 15, "bold"),
        bg=BG, fg=ACCENT
    ).pack(side="left")

    # ── Helper: make a labelled entry ────────────────────────
    def make_entry(parent, label, default, width=12):
        row = tk.Frame(parent, bg=BG, pady=5)
        row.pack(fill="x")
        tk.Label(
            row, text=label,
            font=FONT_L, bg=BG, fg=FG,
            anchor="w", width=26
        ).pack(side="left")
        e = tk.Entry(
            row,
            font=FONT_E,
            bg=ENTRY_BG, fg=ACCENT,
            insertbackground=ACCENT,
            relief="flat", bd=0,
            highlightthickness=1,
            highlightcolor=ACCENT,
            highlightbackground="#2a3a50",
            width=width
        )
        e.insert(0, default)
        e.pack(side="left", padx=(8, 0))
        return e

    entries = {}

    # ── Section: System parameters ───────────────────────────
    def section_label(text):
        f = tk.Frame(root, bg=BG, padx=24)
        f.pack(fill="x", pady=(10, 0))
        tk.Label(
            f, text=text,
            font=("Courier New", 10, "bold"),
            bg=BG, fg="#7ecfef"
        ).pack(anchor="w")

    section_label("ПАРАМЕТРЫ СИСТЕМЫ")
    sys_frame = tk.Frame(root, bg=BG, padx=24)
    sys_frame.pack(fill="x")

    sys_left  = tk.Frame(sys_frame, bg=BG)
    sys_right = tk.Frame(sys_frame, bg=BG)
    sys_left.pack(side="left", padx=(0, 30))
    sys_right.pack(side="left")

    # left column
    def me_l(label, default):
        row = tk.Frame(sys_left, bg=BG, pady=5)
        row.pack(fill="x")
        tk.Label(row, text=label, font=FONT_L, bg=BG, fg=FG,
                 anchor="w", width=26).pack(side="left")
        e = tk.Entry(row, font=FONT_E, bg=ENTRY_BG, fg=ACCENT,
                     insertbackground=ACCENT, relief="flat", bd=0,
                     highlightthickness=1, highlightcolor=ACCENT,
                     highlightbackground="#2a3a50", width=12)
        e.insert(0, default)
        e.pack(side="left", padx=(8, 0))
        return e

    entries["R"]   = me_l("Коэффициент восстановления", "0.22")
    entries["mu"]  = me_l("Амплитуда вибрации μ",       "0.1")
    entries["lam"] = me_l("Частота λ",                   "1.0")

    # right column
    def me_r(label, default):
        row = tk.Frame(sys_right, bg=BG, pady=5)
        row.pack(fill="x")
        tk.Label(row, text=label, font=FONT_L, bg=BG, fg=FG,
                 anchor="w", width=22).pack(side="left")
        e = tk.Entry(row, font=FONT_E, bg=ENTRY_BG, fg=ACCENT,
                     insertbackground=ACCENT, relief="flat", bd=0,
                     highlightthickness=1, highlightcolor=ACCENT,
                     highlightbackground="#2a3a50", width=12)
        e.insert(0, default)
        e.pack(side="left", padx=(8, 0))
        return e

    entries["p_start"]     = me_r("p  начало",           "0.1")
    entries["p_end"]       = me_r("p  конец",            "1.1")
    entries["p_step"]      = me_r("Шаг p",               "0.0002")
    entries["N_transient"] = me_r("Переходных ударов",   "200")
    entries["N_keep"]      = me_r("Записываемых ударов", "1000")

    # ── Divider ──────────────────────────────────────────────
    tk.Frame(root, bg="#2a3a50", height=1).pack(fill="x", padx=24, pady=(12, 0))

    # ── Section: Initial conditions ──────────────────────────
    section_label("НАЧАЛЬНЫЕ УСЛОВИЯ")
    ic_frame = tk.Frame(root, bg=BG, padx=24, pady=2)
    ic_frame.pack(fill="x")

    ic_row = tk.Frame(ic_frame, bg=BG)
    ic_row.pack(fill="x", pady=(4, 0))

    ic_entries = {}

    for key, label, default in [
        ("ic_tau0", "τ₀", "0.0"),
        ("ic_x0",   "x₀", "auto"),
        ("ic_v0",   "v₀", "0.5"),
    ]:
        cell = tk.Frame(ic_row, bg=BG)
        cell.pack(side="left", padx=(0, 28))
        tk.Label(cell, text=label, font=FONT_L, bg=BG, fg=FG,
                 width=3, anchor="w").pack(side="left")
        e = tk.Entry(cell, font=FONT_E, bg=ENTRY_BG, fg=ACCENT,
                     insertbackground=ACCENT, relief="flat", bd=0,
                     highlightthickness=1, highlightcolor=ACCENT,
                     highlightbackground="#2a3a50", width=10)
        e.insert(0, default)
        e.pack(side="left", padx=(4, 0))
        ic_entries[key] = e

    tk.Label(
        ic_frame,
        text='  x₀ = "auto"  →  x₀ = −μ·cos(τ₀)',
        font=("Courier New", 9),
        bg=BG, fg="#5a7a8a"
    ).pack(anchor="w", pady=(6, 0))

    # ── Divider ──────────────────────────────────────────────
    tk.Frame(root, bg="#2a3a50", height=1).pack(fill="x", padx=24, pady=12)

    # ── Run button ───────────────────────────────────────────
    def on_run():
        try:
            params = {k: float(entries[k].get()) for k in entries}
            params['N_transient'] = int(params['N_transient'])
            params['N_keep']      = int(params['N_keep'])
        except ValueError as exc:
            messagebox.showerror("Ошибка ввода", f"Неверное значение:\n{exc}")
            return

        try:
            params['ic_tau0'] = float(ic_entries['ic_tau0'].get())
            params['ic_v0']   = float(ic_entries['ic_v0'].get())
        except ValueError as exc:
            messagebox.showerror("Ошибка ввода", f"Неверное значение τ₀ или v₀:\n{exc}")
            return

        x0_raw = ic_entries['ic_x0'].get().strip().lower()
        if x0_raw == "auto":
            params['ic_x0_auto'] = True
            params['ic_x0']      = None
        else:
            try:
                params['ic_x0_auto'] = False
                params['ic_x0']      = float(x0_raw)
            except ValueError:
                messagebox.showerror("Ошибка ввода", 'x₀ должен быть числом или "auto"')
                return

        if params['p_start'] >= params['p_end']:
            messagebox.showerror("Ошибка", "p_start должен быть меньше p_end")
            return
        if params['p_step'] <= 0:
            messagebox.showerror("Ошибка", "Шаг p должен быть > 0")
            return

        btn.config(state="disabled", text="  Вычисление…  ")
        root.update()
        try:
            run_simulation(params)
        finally:
            btn.config(state="normal", text="  ▶  Построить диаграмму  ")

    btn_frame = tk.Frame(root, bg=BG, pady=4, padx=24)
    btn_frame.pack(fill="x")

    btn = tk.Button(
        btn_frame,
        text="  ▶  Построить диаграмму  ",
        font=("Courier New", 12, "bold"),
        bg=ACCENT, fg="#0d1117",
        activebackground="#81d4fa",
        activeforeground="#0d1117",
        relief="flat", bd=0,
        padx=18, pady=8,
        cursor="hand2",
        command=on_run
    )
    btn.pack(pady=(0, 16))

    # ── Footer ───────────────────────────────────────────────
    tk.Frame(root, bg=ACCENT, height=3).pack(fill="x", side="bottom")

    root.mainloop()


if __name__ == "__main__":
    _R = _mu = _lam = _a = None
    build_gui()
