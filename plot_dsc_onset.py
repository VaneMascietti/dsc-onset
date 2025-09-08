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
try:
    from scipy.signal import savgol_filter as _savgol
except Exception:
    _savgol = None


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
    # Integración del exceso de flujo de calor desde el onset hasta r
    # Usar np.trapz para compatibilidad con NumPy antiguos (equivalente a trapezoid)
    area = float(np.trapz(rel[mask], time[mask]) * 60.0)  # J g^-1
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


# ---------- Suavizado robusto (minutos) ----------
def _smooth_minutes(time, y, win_min: float) -> np.ndarray:
    """Suaviza 'y' en ventanas ~win_min (min), conservando forma del frente.
    Usa Savitzky-Golay si está disponible; si no, media móvil.
    """
    if win_min <= 0:
        return y
    dt = np.median(np.diff(time)) if len(time) > 1 else 0.01
    if dt <= 0:
        return y
    w = int(round(win_min / dt))
    w = max(5, w | 1)  # impar y >=5
    if _savgol is not None and w >= 7:
        return _savgol(y, window_length=w, polyorder=2, mode="interp")
    pad = w // 2
    yy = np.pad(y, (pad, pad), mode="edge")
    ker = np.ones(w) / w
    return np.convolve(yy, ker, mode="valid")


# ---------- Detección de onset por método de la tangente ----------
def detect_onset_first_cross(
    time, hf, *, endo: bool,
    tmin=None, tmax=None,
    smooth_min: float = 0.25,
    base_pre_min: float = 1.5,
    base_gap_min: float = 0.3,
    sigma_k: float = 3.0,
    consec: int = 5,
    edge_halfwin_pts: int = 8,
):
    """
    Devuelve: t_on (min), idx_on, (mb, bb)=base, (mt, bt)=tangente.
    """
    mask = np.ones_like(time, dtype=bool)
    if tmin is not None:
        mask &= (time >= tmin)
    if tmax is not None:
        mask &= (time <= tmax)
    t = time[mask]
    y = hf[mask]
    if t.size < 5:
        idx_on = int(np.argmax(-y if endo else y))
        t_on = float(t[idx_on])
        return t_on, int(np.clip(np.searchsorted(time, t_on), 1, len(time)-1)), (0.0, float(y[0])), (0.0, float(y[0]))

    ys = _smooth_minutes(t, y, smooth_min)
    dydt = np.gradient(ys, t)

    # frente provisional
    k_front = int(np.argmin(dydt) if endo else np.argmax(dydt))
    t_star = float(t[k_front])

    # base antes del frente
    t1 = t_star - float(base_gap_min)
    t0 = t1 - float(base_pre_min)
    base_mask = (t >= t0) & (t <= t1)
    if base_mask.sum() < 8:
        t0 = max(float(t.min()), t0 - 0.5*float(base_pre_min))
        base_mask = (t >= t0) & (t <= t1)
    mb, bb = np.polyfit(t[base_mask], y[base_mask], 1)

    # umbral robusto desde la base
    resid_base = (ys - (mb*t + bb))[base_mask]
    med = float(np.median(resid_base))
    mad = float(np.median(np.abs(resid_base - med))) + 1e-12
    sigma = 1.4826 * mad
    thr = float(sigma_k) * sigma

    s = -1.0 if endo else 1.0
    j0 = int(np.searchsorted(t, t1))
    j_on = None
    for j in range(j0, len(t) - consec):
        seg = s * (ys[j:j+consec] - (mb*t[j:j+consec] + bb))
        good = (seg > thr).all()
        slope_ok = (s * dydt[j:j+consec] > 0).all()
        if good and slope_ok:
            j_on = j
            break
    if j_on is None:
        j_on = k_front

    # tangente local para refinar
    i0 = max(0, j_on - edge_halfwin_pts)
    i1 = min(len(t)-1, j_on + edge_halfwin_pts)
    mt, bt = np.polyfit(t[i0:i1+1], ys[i0:i1+1], 1)

    denom = (mb - mt)
    t_on = float(t[j_on]) if denom == 0 else float((bt - bb) / denom)
    if not (float(t.min()) - 1e-6 <= t_on <= float(t.max()) + 1e-6):
        t_on = float(t[j_on])

    idx_on = int(np.clip(np.searchsorted(time, t_on), 1, len(time)-1))
    return float(t_on), idx_on, (float(mb), float(bb)), (float(mt), float(bt))

def plot_dsc(df: pd.DataFrame, outstem: Path, use_points: bool, endo: bool,
             xlim=None, title: str | None = None, legend_state: str = "on",
             topline: bool = False, window=None, smooth: float = 0.1,
             width_in: float = 3.33, height_in: float = 2.6, dpi: int = 600,
             use_hatch: bool = False, edge_points: int = 12,
             edges: bool = False, area_state: str = "off",
             base_pre: float = 1.5, base_gap: float = 0.3,
             sigma_k: float = 3.0, consec: int = 5, edge_halfwin: int = 8,
             show_time: bool = False):

    time = df["time_min"].to_numpy()
    temp = df["temp_c"].to_numpy()
    hf   = df["hf_wpg"].to_numpy()

    # Detección de onset por tangente ∩ base
    tmin = window[0] if (window and len(window) == 2) else None
    tmax = window[1] if (window and len(window) == 2) else None
    if tmin is not None and tmax is not None:
        n_in_win = int(np.sum((time >= tmin) & (time <= tmax)))
        if n_in_win < 200:
            print(f"[warn] ventana --window contiene {n_in_win} puntos (<200)")

    t_on, idx_on, (mb, bb), (m_edge, b_edge) = detect_onset_first_cross(
        time, hf, endo=endo,
        tmin=tmin, tmax=tmax,
        smooth_min=smooth,
        base_pre_min=base_pre, base_gap_min=base_gap,
        sigma_k=sigma_k, consec=consec, edge_halfwin_pts=edge_halfwin,
    )
    base = mb*time + bb
    T_on = float(np.interp(t_on, time, temp))

    # ΔU desde t_on hasta el final
    r = len(time) - 1
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

    # área (área solo si area_state == 'on')
    if area_state == "on":
        ax.fill_between(time, hf, base,
                        where=(time >= t_on),
                        color="#9dc7e0", alpha=0.30, label=r"Area for $\Delta U$")

    # (se elimina duplicación de trazos para evitar superposición)

    # --- Onset: marcador discreto (sin línea vertical)
    ax.plot([t_on], [hf[idx_on]], marker='o', ms=6, mfc='none', mec='#2ca25f', mew=1.2,
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

    # Con --edges activado, dibujar baseline y tangente en todo el panel
    if edges:
        x0, x1 = ax.get_xlim()
        y0 = m_edge * x0 + b_edge
        y1 = m_edge * x1 + b_edge
        ax.plot(time, base, linestyle="--", linewidth=1.6, color="#6b717e", alpha=0.9,
                label="Baseline line")
        ax.plot([x0, x1], [y0, y1], linestyle=(0, (3, 1)), linewidth=1.8, color="#8a8f99",
                label="Edge tangent")

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
    # marcador repetido para mantener el estilo existente de la etiqueta
    ax.plot([t_on], [hf[idx_on]], marker="o", ms=6, mfc="none", mec="#2ca25f", mew=1.2,
            zorder=6, linewidth=0, label="Onset")
    # y_on ahora se calcula sobre la base en t_on para no cambiar la posición visual
    y_on = (mb * t_on + bb)
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
    p.add_argument("--smooth", type=float, default=0.25, metavar="MIN",
                   help="Suavizado (min) para derivada/residuo")
    p.add_argument("--base-pre", type=float, default=1.5,
                   help="Min previos al frente para ajustar línea base")
    p.add_argument("--base-gap", type=float, default=0.3,
                   help="Separación (min) entre base y frente")
    p.add_argument("--sigma", type=float, default=3.0,
                   help="Umbral robusto en múltiplos de sigma (MAD)")
    p.add_argument("--consec", type=int, default=5,
                   help="Puntos consecutivos sobre umbral para confirmar onset")
    p.add_argument("--edge-halfwin", type=int, default=8,
                   help="Semiventana (puntos) para ajustar la tangente local")
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
        base_pre=args.base_pre,
        base_gap=args.base_gap,
        sigma_k=args.sigma,
        consec=args.consec,
        edge_halfwin=args.edge_halfwin,
    )
    print(f"Usando archivo: {args.file.name}")
    print(f"Onset: {T_on:.2f} °C (t={t_on:.2f} min)   ΔU={dU:.3f} J g^-1")
    print(f"Exportado: {outstem.name}.png / .pdf / .svg")


if __name__ == "__main__":
    main()
