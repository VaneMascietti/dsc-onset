#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot DSC (TRIOS) con onset y ΔU — compatible Linux/WSL/Windows/Mac
- Lee TXT exportado de TRIOS (1ª fila nombres, 2ª fila unidades).
- Eje X: Time (min). Eje Y izq: Heat flow (W g^-1). Eje Y der: Temperature (°C).
- Detecta el primer pico “grande” y calcula ΔU con baseline lineal.
- Guarda PNG, PDF y SVG listos para revista (sin cuadrícula, leyendas afuera).

Uso:
  python plot_dsc_onset.py --file 240614_metoh_nc5_0.111w_0.2rate.txt
Opciones:
  --points         plotea puntos en vez de línea
  --endo           trata el pico principal como endotérmico (signo ΔU negativo)
  --xlim 25 120    limita el rango de tiempo (min min)
  --title " "      fija título; por omisión no coloca título
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
# Asegura backend sin display (Linux/WSL/CI)
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------- Lectura robusta de TRIOS ----------
def load_trios_txt(path: Path) -> pd.DataFrame:
    """
    Lee TXT de TRIOS exportado como tabla.
    Espera cabecera con nombres y segunda fila con unidades.
    Devuelve DataFrame con columnas: time (min), temp (°C), hf (W/g).
    """
    df = pd.read_csv(
        path, sep=r"\t", engine="python",
        header=0, skiprows=[1],  # salta la fila de unidades
        comment="#", dtype=str
    )

    # Normaliza nombres
    cols = {c: c.lower().strip() for c in df.columns}
    df.rename(columns=cols, inplace=True)

    # Detecta las columnas por palabras clave
    def col_like(*keys):
        cand = [c for c in df.columns if all(k in c for k in keys)]
        return cand[0] if cand else None

    c_time = col_like("time")
    c_temp = col_like("temp") or col_like("°c")
    c_hf   = col_like("heat", "flow") or col_like("normalized")

    if not all([c_time, c_temp, c_hf]):
        raise ValueError(
            f"No pude encontrar columnas necesarias en {path.name}.\n"
            f"Detectadas: {list(df.columns)}"
        )

    # A numérico (respeta punto decimal)
    for c in [c_time, c_temp, c_hf]:
        df[c] = pd.to_numeric(df[c].str.replace(",", ".", regex=False), errors="coerce")

    df = df.dropna(subset=[c_time, c_temp, c_hf]).reset_index(drop=True)
    df = df.rename(columns={c_time: "time_min", c_temp: "temp_c", c_hf: "hf_wpg"})
    return df[["time_min", "temp_c", "hf_wpg"]]


# ---------- Detección de pico y baseline ----------
def find_peak_and_window(time, hf, t_hint_min: float | None = None):
    """
    Encuentra un “primer” pico grande por cruce de umbral sobre una
    señal suavizada/centrada. Devuelve índices (l, peak, r) de la ventana.
    """
    # Quita offset local para que el umbral sea estable
    # usa una franja previa amplia
    baseline_level = np.median(hf[np.logical_and(time >= time.min() + 2, time <= time.min() + 6)])
    sig = hf - baseline_level

    # Suavizado simple (mediana móvil)
    k = max(5, int(0.1 / np.median(np.diff(time))))  # ~0.1 min
    k += (k + 1) % 2  # impar
    pad = k // 2
    padded = np.pad(sig, (pad, pad), mode="edge")
    smooth = np.convolve(padded, np.ones(k) / k, mode="valid")

    # Umbral adaptativo
    region = np.logical_and(time >= time.min() + 5, time <= time.min() + 40)
    thr = np.maximum(2.0 * np.std(smooth[region]), 0.0002)

    idx = np.where(smooth > thr)[0]

    # Agrupa índices contiguos y elige el primer grupo largo
    runs = []
    if idx.size:
        s = p = idx[0]
        for i in idx[1:]:
            if i == p + 1:
                p = i
            else:
                runs.append((s, p))
                s = p = i
        runs.append((s, p))

    if not runs:
        # fallback: máximo absoluto en todo el registro
        peak = np.argmax(hf)
        l = max(0, peak - 30)
        r = min(len(hf) - 1, peak + 30)
        return l, peak, r

    l0, r0 = runs[0]
    peak = l0 + np.argmax(hf[l0:r0 + 1])

    # expandir ventana hasta que la señal caiga a casi-baseline
    eps = max(0.3 * thr, 1e-5)
    l = l0
    r = r0
    while l > 0 and (smooth[l - 1] > eps):
        l -= 1
    while r < len(smooth) - 1 and (smooth[r + 1] > eps):
        r += 1

    return l, peak, r


def baseline_linear(time, hf, l, r):
    """Recta entre (l) y (r)."""
    if time[r] == time[l]:
        m = 0.0
    else:
        m = (hf[r] - hf[l]) / (time[r] - time[l])
    b = hf[l] - m * time[l]
    return m * time + b


# ---------- Gráfico estilo “journal” ----------
def plot_dsc(df: pd.DataFrame, outstem: Path, use_points: bool, endo: bool,
             xlim=None, title: str | None = None):

    time = df["time_min"].to_numpy()
    temp = df["temp_c"].to_numpy()
    hf   = df["hf_wpg"].to_numpy()

    l, peak, r = find_peak_and_window(time, hf)

    base = baseline_linear(time, hf, l, r)
    mask = (time >= time[l]) & (time <= time[r])

    # ΔU (J g^-1). usar numpy.trapz (compatibilidad amplia)
    area = np.trapz(hf[mask] - base[mask], time[mask]) * 60.0
    if endo:
        area = -abs(area)  # convención endotérmica negativa
    t_on = time[l]
    T_on = np.interp(t_on, time, temp)

    # Estilo general (sin grid, tipografías sobrias)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": False,
    })

    fig, ax = plt.subplots(figsize=(6.0, 3.6), dpi=300)

    # Heat flow
    if use_points:
        ax.plot(time, hf, linestyle="none", marker="o", markersize=2.5,
                alpha=0.9, color="#cc6d1c", label="Heat flow")
    else:
        ax.plot(time, hf, linewidth=2.0, color="#cc6d1c", label="Heat flow")

    # Baseline y área ΔU
    ax.plot(time[l:r+1], base[l:r+1], linestyle="--", linewidth=1.8,
            color="#6b717e", label="Baseline")
    ax.fill_between(time[l:r+1], hf[l:r+1], base[l:r+1],
                    color="#9dc7e0", alpha=0.35, label=r"Area for $\Delta U$")

    # Onset
    ax.axvline(t_on, color="#2ca25f", linestyle=":", linewidth=1.5)
    ax.plot([t_on], [base[l]], marker="*", markersize=10, color="#2ca25f",
            label="Onset")

    ax.set_xlabel("Time (min)")
    ax.set_ylabel(r"Heat flow (W g$^{-1}$)")

    # Temperatura (eje derecho)
    ax2 = ax.twinx()
    ax2.plot(time, temp, color="#33475b", linestyle="--", linewidth=1.6, label="Temperature")
    ax2.set_ylabel("Temperature (°C)")

    # Límites
    if xlim and len(xlim) == 2:
        ax.set_xlim(xlim)
        ax2.set_xlim(xlim)
    else:
        ax.set_xlim(time.min(), time.max())
        ax2.set_xlim(time.min(), time.max())

    # Quita marcos superiores
    for a in (ax, ax2):
        a.spines["top"].set_visible(False)

    if title:
        ax.set_title(title)

    # Leyenda fuera (abajo), sin tapar la curva
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.08), frameon=False)

    # Etiqueta con resultados (caja discreta)
    ax.annotate(
        f"Onset: {T_on:.2f} °C (t = {t_on:.2f} min)\n"
        r"$\Delta U$: " + f"{area:.3f} J g$^{{-1}}$",
        xy=(t_on, base[l]),
        xytext=(t_on + 0.04*(time.max()-time.min()), base[l] + 0.2*(hf.max()-hf.min())),
        fontsize=10,
        color="#1b1b1b",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbbbbb", alpha=0.9),
        arrowprops=dict(arrowstyle="-|>", color="#2ca25f", lw=1.0)
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    outstem = outstem.with_suffix("")  # quita extensión si viene con una
    for ext in (".png", ".pdf", ".svg"):
        fig.savefig(outstem.as_posix() + ext, bbox_inches="tight")
    plt.close(fig)

    return T_on, t_on, area


# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Plot DSC con onset y ΔU (TRIOS TXT)")
    p.add_argument("--file", required=True, type=Path, help="Ruta al TXT exportado por TRIOS")
    p.add_argument("--points", action="store_true", help="Graficar puntos en vez de línea")
    p.add_argument("--endo", action="store_true", help="Convención endotérmica (ΔU<0)")
    p.add_argument("--xlim", nargs=2, type=float, metavar=("XMIN", "XMAX"))
    p.add_argument("--title", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    df = load_trios_txt(args.file)
    outstem = args.file.with_name(args.file.stem + "_publication")
    T_on, t_on, dU = plot_dsc(df, outstem, args.points, args.endo, args.xlim, args.title)
    print(f"Usando archivo: {args.file.name}")
    print(f"Onset: {T_on:.2f} °C (t={t_on:.2f} min)   ΔU={dU:.3f} J g^-1")
    print(f"Exportado: {outstem.name}.png / .pdf / .svg")


if __name__ == "__main__":
    main()
