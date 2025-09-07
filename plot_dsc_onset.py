#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot DSC (TRIOS) con onset y ΔU — compatible Linux/WSL/Windows/Mac

- Lee TXT exportado de TRIOS (1ª fila nombres, 2ª fila unidades).
- Eje X: Time (min). Eje Y izq: Heat flow (W g^-1). Eje Y der: Temperature (°C).
- Detecta el primer pico “grande”, calcula ΔU con baseline lineal y onset por cruce
  de baseline con la tangente de la subida del pico.
- Salidas PNG, PDF, SVG listas para revista (sin grid; leyendas afuera/optativas).

Uso:
  python plot_dsc_onset.py --file 240614_metoh_nc5_0.111w_0.2rate.txt
Opciones:
  --points            graficar puntos en vez de línea
  --endo              convención endotérmica (ΔU<0)
  --xlim 25 120       limitar rango de tiempo (min min)
  --title "..."       título (opcional)
  --legend off        no mostrar leyenda
  --topline           dibuja línea superior del marco (sin ticks/labels)
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe


# ---------- Lectura robusta de TRIOS ----------
def load_trios_txt(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path, sep=r"\t", engine="python",
        header=0, skiprows=[1], comment="#", dtype=str
    )
    cols = {c: c.lower().strip() for c in df.columns}
    df.rename(columns=cols, inplace=True)

    def col_like(*keys):
        cand = [c for c in df.columns if all(k in c for k in keys)]
        return cand[0] if cand else None

    c_time = col_like("time")
    c_temp = col_like("temp") or col_like("°c")
    c_hf   = col_like("heat", "flow") or col_like("normalized")
    if not all([c_time, c_temp, c_hf]):
        raise ValueError(f"Columnas no encontradas en {path.name}.\nDetectadas: {list(df.columns)}")

    for c in (c_time, c_temp, c_hf):
        df[c] = pd.to_numeric(df[c].str.replace(",", ".", regex=False), errors="coerce")
    df = df.dropna(subset=[c_time, c_temp, c_hf]).reset_index(drop=True)
    df = df.rename(columns={c_time: "time_min", c_temp: "temp_c", c_hf: "hf_wpg"})
    return df[["time_min", "temp_c", "hf_wpg"]]


# ---------- Detección de pico & baseline ----------
def find_peak_and_window(time, hf, endo: bool = False, smooth_width_min: float = 0.1):
    base_lvl = np.median(hf[(time >= time.min()+2) & (time <= time.min()+6)])
    sig = hf - base_lvl
    # Suavizado: media móvil controlada por ancho en minutos
    dt = np.median(np.diff(time)) if len(time) > 1 else 0.01
    k = max(5, int(max(smooth_width_min, dt) / max(dt, 1e-6)))
    if k % 2 == 0:
        k += 1
    pad = k // 2
    smooth = np.convolve(np.pad(sig, (pad, pad), "edge"), np.ones(k)/k, mode="valid")

    # Detección con signo coherente: exo arriba (endo=False) o endo abajo (endo=True)
    sgn = -1.0 if endo else 1.0
    thr = max(1.5*np.std(sgn*smooth), 2e-4)
    idx = np.where(sgn*smooth > thr)[0]

    if idx.size == 0:
        peak = int(np.argmax(hf))
        l = max(0, peak-30); r = min(len(hf)-1, peak+30)
        return l, peak, r

    # agrupar índices consecutivos y elegir el grupo con mayor amplitud
    runs, s, p = [], idx[0], idx[0]
    for i in idx[1:]:
        if i == p+1: p = i
        else: runs.append((s, p)); s = p = i
    runs.append((s, p))
    # elegir run con máximo pico en sgn*smooth
    best = max(runs, key=lambda ab: float(np.max(sgn*smooth[ab[0]:ab[1]+1])))
    l0, r0 = best
    if endo:
        peak = l0 + int(np.argmin(hf[l0:r0+1]))
    else:
        peak = l0 + int(np.argmax(hf[l0:r0+1]))
    # expandir hasta casi-baseline
    eps = max(0.3*thr, 1e-5)
    l, r = l0, r0
    while l>0 and sgn*smooth[l-1] > eps: l -= 1
    while r < len(smooth)-1 and sgn*smooth[r+1] > eps: r += 1
    return l, peak, r


def baseline_linear(time, hf, l, r):
    if time[r] == time[l]:
        m = 0.0
    else:
        m = (hf[r]-hf[l])/(time[r]-time[l])
    b = hf[l] - m*time[l]
    return m*time + b, m, b


# ---------- Onset y Área mejorados ----------
def refine_onset_by_tangent(time, hf, base, l, peak, r, endo: bool = False,
                            edge_points: int = 12):
    """
    Onset = cruce entre baseline y recta ajustada a la subida del pico.
    Selecciono puntos del borde ascendente (entre ~10% y 60% de la amplitud).
    """
    # amplitud local respecto de baseline
    # Derivada en el tramo ascendente (leading-edge)
    sgn = -1.0 if endo else 1.0
    rng_start, rng_end = (l, peak) if peak >= l else (peak, l)
    rel = sgn * (hf - base)
    if rng_end - rng_start < 4:
        return time[l], base[l], 0.0, base[l], float(time[l]), float(time[l])
    d_rel = np.gradient(rel[rng_start:rng_end+1], time[rng_start:rng_end+1])
    imax_rel = int(np.argmax(d_rel)) + rng_start
    # Ventana de ±edge_points alrededor de imax dentro de [l, peak]
    i0 = max(l, imax_rel - edge_points)
    i1 = min(peak, imax_rel + edge_points)
    if i1 - i0 < 2:
        return time[l], base[l], 0.0, base[l], float(time[l]), float(time[l])
    jj = np.arange(i0, i1 + 1)

    # ajuste lineal a hf (no al rel), así intersecto con baseline en el mismo espacio
    t_fit = time[jj]
    y_fit = hf[jj]
    m_edge, b_edge = np.polyfit(t_fit, y_fit, 1)

    # baseline local ~ lineal; aproximamos con recta por (l,r)
    m_base = (hf[r]-hf[l])/(time[r]-time[l]) if time[r]!=time[l] else 0.0
    b_base = hf[l] - m_base*time[l]

    if abs(m_edge - m_base) < 1e-12:
        t_on = time[i0]
    else:
        t_on = (b_base - b_edge) / (m_edge - m_base)
    if not (time[l]-1.0 <= t_on <= time[r]+1.0):
        t_on = time[l]
    # recorte adicional al dominio [time[l], time[peak]]
    t_on = min(max(t_on, time[l]), time[peak] if peak >= l else time[l])
    y_on = m_base*t_on + b_base
    return float(t_on), float(y_on), m_edge, b_edge, float(t_fit.min()), float(t_fit.max())


def integrate_area_from_onset(time, hf, base, t_on, r, endo=False):
    rel = hf - base
    mask = (time >= t_on) & (time <= time[r])
    if np.sum(mask) < 2:
        return 0.0
    area = float(np.trapezoid(rel[mask], time[mask]) * 60.0)  # J g^-1
    return -abs(area) if endo else abs(area)


# ---------- Gráfico estilo “journal” ----------
def _moving_average(time: np.ndarray, y: np.ndarray, width_min: float) -> np.ndarray:
    dt = np.median(np.diff(time)) if len(time) > 1 else 0.01
    k = max(5, int(max(width_min, dt) / max(dt, 1e-6)))
    if k % 2 == 0:
        k += 1
    pad = k // 2
    return np.convolve(np.pad(y, (pad, pad), "edge"), np.ones(k)/k, mode="valid")


# --- NUEVO: helper para ubicar etiqueta sin cruzar la rampa ---
def _label_offset_auto(time, temp, t_on):
    """Devuelve (dx_pts, dy_pts, ha) para anotar sin cruzar la rampa.
    Si la rampa tiene pendiente negativa, coloco la etiqueta arriba-izquierda;
    si positiva, arriba-derecha. Offsets en puntos (pt)."""
    # ventana local ±4 min alrededor del onset
    w = 4.0
    sel = (time >= t_on - w) & (time <= t_on + w)
    if np.sum(sel) >= 2:
        m = np.polyfit(time[sel], temp[sel], 1)[0]
    else:
        m = 1.0
    if m < 0:   # rampa descendente -> etiqueta a la izquierda
        return (-50, 22, "right")
    else:       # rampa ascendente -> etiqueta a la derecha
        return (50, 22, "left")


def format_du(x):
    if abs(x) < 5e-4:
        x = 0.0
    return f"{x:.3f}"

# Halo blanco para texto legible sin caja
PE_HALO = [pe.withStroke(linewidth=3, foreground="white", alpha=0.9)]


def plot_dsc(df: pd.DataFrame, outstem: Path, use_points: bool, endo: bool,
             xlim=None, title: str | None = None, legend_state: str = "on",
             topline: bool = False, window=None, smooth: float = 0.1,
             width_in: float = 3.33, height_in: float = 2.6, dpi: int = 600,
             use_hatch: bool = False, edge_points: int = 12,
             edges: bool = False, area_state: str = "off"):

    time = df["time_min"].to_numpy()
    temp = df["temp_c"].to_numpy()
    hf   = df["hf_wpg"].to_numpy()

    # Ventana de búsqueda del pico: automática o forzada por CLI
    if window and len(window) == 2:
        wmask = (time >= window[0]) & (time <= window[1])
        if np.any(wmask):
            l_loc, peak_loc, r_loc = find_peak_and_window(time[wmask], hf[wmask], endo=endo, smooth_width_min=smooth)
            idxs = np.where(wmask)[0]
            l, peak, r = idxs[l_loc], idxs[peak_loc], idxs[r_loc]
        else:
            l, peak, r = find_peak_and_window(time, hf, endo=endo, smooth_width_min=smooth)
    else:
        l, peak, r = find_peak_and_window(time, hf, endo=endo, smooth_width_min=smooth)
    base, mb, bb = baseline_linear(time, hf, l, r)

    # onset por leading-edge vs baseline
    t_on, y_on, m_edge, b_edge, t0_fit, t1_fit = refine_onset_by_tangent(
        time, hf, base, l, peak, r, endo=endo, edge_points=edge_points
    )
    T_on = float(np.interp(t_on, time, temp))

    # ΔU desde t_on hasta r
    area = integrate_area_from_onset(time, hf, base, t_on, r, endo=endo)

    # estilo
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.grid": False,
    })
    fig, ax = plt.subplots(figsize=(width_in or 3.33, height_in or 2.6), dpi=(dpi or 600))

    # heat flow
    if use_points:
        ax.plot(time, hf, ls="none", marker="o", ms=2.8, alpha=0.95,
                color="#1f77b4", label="Heat flow")
    else:
        ax.plot(time, hf, lw=2.1, color="#1f77b4", label="Heat flow")

    # baseline & área (área solo si area_state == 'on')
    ax.plot(time[l:r+1], base[l:r+1], ls="--", lw=1.7, color="#6b717e", label="Baseline")
    if area_state == "on":
        ax.fill_between(time[l:r+1], hf[l:r+1], base[l:r+1],
                        where=(time[l:r+1] >= t_on),
                        color="#9dc7e0", alpha=0.30, label=r"Area for $\Delta U$")

    # Rectas auxiliares opcionales: baseline ya está; tangente si --edges
    if edges:
        tt = np.array([t0_fit, t1_fit])
        yy = m_edge*tt + b_edge
        ax.plot(tt, yy, lw=1.8, color="#555", alpha=0.9, label="Edge tangent", zorder=4)

    # (se elimina duplicación de trazos para evitar superposición)

    # --- Onset: marcador discreto (sin línea vertical)
    ax.plot([t_on], [y_on], marker='o', ms=6, mfc='none', mec='#2ca25f', mew=1.2,
            linewidth=0, zorder=6, label="Onset")

    ax.set_xlabel("Time (min)")
    ax.set_ylabel(r"Heat flow (W g$^{-1}$)")

    # Temperatura (eje derecho, rampa sobria)
    ax2 = ax.twinx()
    ax2.plot(time, temp, ls="--", lw=1.2, color="#6b717e", alpha=0.7, zorder=1, label="Temperature")
    ax2.set_ylabel("Temperature (°C)")
    ax2.tick_params(axis="y", which="both", length=4, width=0.8)

    # límites
    if xlim and len(xlim) == 2:
        ax.set_xlim(xlim); ax2.set_xlim(xlim)
    else:
        ax.set_xlim(time.min(), time.max()); ax2.set_xlim(time.min(), time.max())

    # Marcos: sin grid, ticks hacia fuera, líneas limpias
    for a in (ax, ax2):
        a.grid(False)
        a.tick_params(direction="out", length=4, width=0.8)
        # NO usar espina superior por defecto (la activamos solo si --topline)
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False if a is ax else True)  # ax2 sí muestra derecha
        a.spines["left"].set_visible(True if a is ax else False)
        a.spines["bottom"].set_visible(True)
        a.spines["left"].set_linewidth(1.1)
        a.spines["bottom"].set_linewidth(1.1)

    # Añade aire arriba para que la etiqueta no tape la curva
    ymin, ymax = ax.get_ylim()
    rng = (ymax - ymin) if (ymax > ymin) else max(abs(ymax), 1.0)
    ax.set_ylim(ymin, ymax + 0.15*rng)

    # Línea superior de cierre (sin números ni ticks) — NO usar secondary_xaxis
    ax.spines["top"].set_visible(bool(topline))
    if topline:
        ax.tick_params(axis="x", which="both", top=False, labeltop=False)
        ax.spines["top"].set_linewidth(ax.spines["bottom"].get_linewidth())

    # No título dentro de la figura (caption va en el manuscrito)

    # Leyenda (opcional). Si está off, no dibujar nada; sin rótulos inline
    if legend_state == "on":
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        handles = h1 + h2
        labels = l1 + l2
        uniq_h, uniq_l = [], []
        for h, l in zip(handles, labels):
            if not l or l.startswith("_"):  # ignora etiquetas vacías/privadas
                continue
            if l in uniq_l:
                continue
            uniq_h.append(h); uniq_l.append(l)
        fig.legend(uniq_h, uniq_l, loc="lower center", ncol=3,
                   bbox_to_anchor=(0.5, -0.12), frameon=False)

    # Onset limpio: marcador y texto fuera del panel (arriba-izquierda)
    txt = (
        f"Onset: {T_on:.2f} °C ({t_on:.2f} min)\n" +
        r"$\Delta U$: " + f"{format_du(area)} J g$^{{-1}}$"
    )
    ax.plot([t_on], [y_on], marker="o", ms=6, mfc="none", mec="#2ca25f", mew=1.2,
            zorder=6, linewidth=0, label="Onset")
    ax.annotate(
        txt,
        xy=(t_on, y_on), xycoords="data",
        xytext=(0.97, 0.985), textcoords="axes fraction",
        ha="right", va="top",
        path_effects=PE_HALO
    )

    # Export listo para revista (PNG alta, y vectoriales)
    if legend_state == 'off':
        plt.tight_layout(rect=[0, 0.02, 1, 1])
    else:
        plt.tight_layout(rect=[0, 0.08, 1, 1])

    outstem = outstem.with_suffix("")
    fig.savefig(outstem.as_posix() + ".png", dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(outstem.as_posix() + ".pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(outstem.as_posix() + ".svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return T_on, float(t_on), float(area)


# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Plot DSC con onset y ΔU (TRIOS TXT)")
    p.add_argument("--file", required=True, type=Path, help="Ruta al TXT exportado por TRIOS")
    p.add_argument("--points", action="store_true", help="Graficar puntos en vez de línea")
    p.add_argument("--endo", action="store_true", help="Convención endotérmica (ΔU<0)")
    p.add_argument("--xlim", nargs=2, type=float, metavar=("XMIN", "XMAX"))
    p.add_argument("--title", type=str, default=None, help="(Ignorado) No colocar título en figura")
    p.add_argument("--legend", choices=["on","off"], default="off",
                   help="Leyenda general fuera del área (default: off)")
    p.add_argument("--topline", action="store_true",
                   help="Cierra el cuadro con línea superior sin ticks")
    p.add_argument("--window", nargs=2, type=float, metavar=("TMIN","TMAX"),
                   help="Fuerza ventana de búsqueda de pico (minutos) para onset/ΔU")
    p.add_argument("--smooth", type=float, default=0.12, metavar="MIN",
                   help="Ancho de suavizado (min) para detección (no se dibuja)")
    p.add_argument("--width-in", "--width_in", dest="width_in", type=float, default=4.5,
                   help="Ancho figura (in, JCED 1-col por defecto)")
    p.add_argument("--height-in", "--height_in", dest="height_in", type=float, default=2.6,
                   help="Alto figura (in, JCED 1-col por defecto)")
    p.add_argument("--dpi", type=int, default=600, help="DPI para PNG")
    p.add_argument("--hatch", action="store_true", help="Usar hatch en ΔU en lugar de alpha")
    p.add_argument("--edge-points", type=int, default=12, help="± puntos para ajustar la tangente")
    p.add_argument("--edges", action="store_true",
                   help="Muestra baseline y tangente del onset")
    p.add_argument("--area", choices=["on","off"], default="off",
                   help="Muestra/oculta el sombreado del área para ΔU")
    return p.parse_args()


def main():
    args = parse_args()
    df = load_trios_txt(args.file)
    outstem = args.file.with_name(args.file.stem + "_publication")
    T_on, t_on, dU = plot_dsc(
        df, outstem,
        use_points=args.points,
        endo=args.endo,
        xlim=args.xlim,
        title=args.title,
        legend_state=args.legend,
        topline=args.topline,
        window=args.window,
        smooth=args.smooth,
        width_in=args.width_in,
        height_in=args.height_in,
        dpi=args.dpi,
        use_hatch=args.hatch,
        edge_points=args.edge_points,
        edges=args.edges,
        area_state=args.area,
    )
    print(f"Usando archivo: {args.file.name}")
    print(f"Onset: {T_on:.2f} °C (t={t_on:.2f} min)   ΔU={dU:.3f} J g^-1")
    print(f"Exportado: {outstem.name}.png / .pdf / .svg")


if __name__ == "__main__":
    main()
