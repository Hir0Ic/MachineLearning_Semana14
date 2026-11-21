"""Evaluación final en prueba, carga de resultados y gráficos de la Parte 4.

Cubre las secciones 5.2 y 5.4 del informe:
- Carga la configuración elegida en validación (`results/selection.json`) y su modelo.
- Evalúa UNA sola vez sobre las 2 607 horas de prueba, junto con los baselines
  calculados sobre exactamente las mismas horas.
- Guarda en `results/`: modelo final, escalador, predicciones y métricas de prueba.
- Expone funciones de lectura y de gráficos que reutilizan la app (pestañas 4 y 5)
  y la generación de figuras del informe.

Uso:
    python evaluate.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from baselines import evaluate_baselines  # noqa: E402
from config import (  # noqa: E402
    DEFAULT_SEASONAL_DAYS,
    FEATURES,
    RESULTS_DIR,
    SEED,
    TARGET,
    TRAIN_FRAC,
    VAL_FRAC,
    set_seeds,
)
from data import load_hour  # noqa: E402
from features import add_cyclic_features  # noqa: E402
from metrics import mae, mase, rmse  # noqa: E402
from model import SimpleRNNForecaster  # noqa: E402
from train import build_window_datasets, predict  # noqa: E402

FIG_DIR = RESULTS_DIR / "figures"
BLUE, ORANGE, GREEN, RED, INK, GREY = "#2B579A", "#C8581A", "#2E7D4F", "#B5251D", "#1F2A44", "#8A94A6"

PERSIST = "A. Persistencia: cnt(t)"
SEASONAL = f"B. Promedio estacional (k={DEFAULT_SEASONAL_DAYS} días)"
SEASONAL_1 = "B0. Misma hora día anterior (k=1 día)"
RIDGE = "C. Regresión Ridge sobre rezagos (opcional)"


# ======================================================================== carga
def results_available(results_dir: Path = RESULTS_DIR) -> Dict[str, bool]:
    """Indica qué etapas de la Parte 4 ya fueron ejecutadas."""
    return {
        "grid": (results_dir / "grid_results.csv").exists() and (results_dir / "selection.json").exists(),
        "test": (results_dir / "test_metrics.csv").exists() and (results_dir / "test_predictions.csv").exists(),
    }


def load_results(results_dir: Path = RESULTS_DIR) -> Dict[str, object]:
    """Lee todos los artefactos guardados por experiments.py y evaluate.py."""
    out: Dict[str, object] = {}
    rd = Path(results_dir)

    def _json(name):
        p = rd / name
        if p.exists():
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        return None

    def _csv(name, **kw):
        p = rd / name
        return pd.read_csv(p, **kw) if p.exists() else None

    out["grid"] = _csv("grid_results.csv")
    out["baselines_val"] = _csv("baselines_val.csv")
    out["val_predictions"] = _csv("val_predictions.csv", parse_dates=["timestamp"])
    out["test_metrics"] = _csv("test_metrics.csv")
    out["test_predictions"] = _csv("test_predictions.csv", parse_dates=["timestamp"])
    out["histories"] = _json("histories.json")
    out["selection"] = _json("selection.json")
    out["environment"] = _json("environment.json")
    out["train_config"] = _json("train_config.json")
    out["scaler"] = _json("scaler.json")
    out["reproducibility"] = _json("reproducibility.json")
    return out


def load_model(checkpoint_path: Path) -> SimpleRNNForecaster:
    """Reconstruye una SimpleRNNForecaster a partir de un checkpoint guardado en results/."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = SimpleRNNForecaster(
        input_size=ckpt["input_size"], hidden_size=ckpt["hidden_size"], num_layers=ckpt["num_layers"]
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


# ======================================================================== evaluación final
def improvement_table(y_true: np.ndarray, preds: Dict[str, np.ndarray], y_train: np.ndarray) -> pd.DataFrame:
    """MAE, RMSE, MASE y mejora porcentual (MAE y RMSE) respecto a la persistencia."""
    p_mae, p_rmse = mae(y_true, preds[PERSIST]), rmse(y_true, preds[PERSIST])
    rows = []
    for name, yhat in preds.items():
        m, r = mae(y_true, yhat), rmse(y_true, yhat)
        rows.append({
            "Modelo": name,
            "MAE": m,
            "RMSE": r,
            "MASE": mase(y_true, yhat, y_train),
            "Mejora MAE vs. persistencia (%)": (1 - m / p_mae) * 100,
            "Mejora RMSE vs. persistencia (%)": (1 - r / p_rmse) * 100,
        })
    return pd.DataFrame(rows)


def evaluate_on_test(results_dir: Path = RESULTS_DIR, verbose: bool = True) -> pd.DataFrame:
    """Evaluación ÚNICA en prueba de la configuración elegida y de los baselines."""
    set_seeds(SEED)
    rd = Path(results_dir)
    with open(rd / "selection.json", encoding="utf-8") as f:
        sel = json.load(f)

    df = add_cyclic_features(load_hour())
    data = build_window_datasets(df=df, L=sel["L"], feature_cols=FEATURES, target_col=TARGET)
    model = load_model(rd / "models" / f"{sel['ID']}.pt")
    y_pred = predict(model, data["X_test"], scaler=data["scaler"], device="cpu")
    y_test = data["y_test_raw"].astype(float)

    _, base_preds, y_test_b, ts_b = evaluate_baselines(
        df, split="test", seasonal_days=DEFAULT_SEASONAL_DAYS, include_lag_reg=True,
        train_frac=TRAIN_FRAC, val_frac=VAL_FRAC,
    )
    assert np.allclose(y_test_b, y_test), "RNN y baselines deben evaluarse sobre las mismas horas"
    assert (ts_b.values == data["test_timestamps"].values).all()

    rnn_name = f"RNN seleccionada ({sel['ID']}: L = {sel['L']}, H = {sel['H']})"
    preds = {
        PERSIST: base_preds[PERSIST],
        SEASONAL: base_preds[SEASONAL],
        SEASONAL_1: base_preds[SEASONAL_1],
        RIDGE: base_preds[RIDGE],
        rnn_name: y_pred,
    }
    table = improvement_table(y_test, preds, data["y_train_series_raw"])
    table.to_csv(rd / "test_metrics.csv", index=False)

    # Predicciones de prueba + contexto para el análisis de error (Parte 5)
    n = len(df)
    i_va = int(n * (TRAIN_FRAC + VAL_FRAC))
    ctx = df.iloc[i_va:].reset_index(drop=True)
    out = pd.DataFrame({
        "timestamp": data["test_timestamps"],
        "hr": ctx["hr"], "weekday": ctx["weekday"], "workingday": ctx["workingday"],
        "holiday": ctx["holiday"], "weathersit": ctx["weathersit"],
        "temp": ctx["temp"], "hum": ctx["hum"], "windspeed": ctx["windspeed"],
        "y_true": y_test,
        "rnn": y_pred,
        "persistencia": base_preds[PERSIST],
        "promedio_estacional": base_preds[SEASONAL],
        "misma_hora_dia_anterior": base_preds[SEASONAL_1],
        "ridge": base_preds[RIDGE],
    })
    out.to_csv(rd / "test_predictions.csv", index=False)

    # Artefactos finales: modelo seleccionado y escalador ajustado solo en Train
    torch.save(torch.load(rd / "models" / f"{sel['ID']}.pt", weights_only=False), rd / "best_model.pt")
    with open(rd / "scaler.json", "w", encoding="utf-8") as f:
        json.dump({"tipo": "MinMax (ajustado solo en entrenamiento)",
                   "features": FEATURES, "target": TARGET,
                   "stats": {k: list(v) for k, v in data["scaler"].stats_.items()}}, f, indent=2)

    if verbose:
        print(table.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    return table


# ======================================================================== gráficos
def _style(ax):
    ax.grid(True, linestyle=":", alpha=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_protocol():
    """Esquema del protocolo de validación por bloques temporales (sección 4.7)."""
    fig, ax = plt.subplots(figsize=(11, 3.0))
    spans = [(0, 70, BLUE, "Entrenamiento (70 %) · 12 165 h", "Ajuste de pesos y del escalador MinMax"),
             (70, 85, ORANGE, "Validación (15 %)\n2 607 h", "Early stopping\ny selección C1–C9"),
             (85, 100, RED, "Prueba (15 %)\n2 607 h", "Una sola evaluación\n(modelo ya elegido)")]
    for a, b, c, txt, use in spans:
        ax.barh(1, b - a, left=a, color=c, alpha=0.88, height=0.55)
        ax.text((a + b) / 2, 1, txt, ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")
        ax.text((a + b) / 2 + (1.5 if a > 0 else 0), 0.5, use, ha="center", va="center", color=c, fontsize=8)
    # Ventana de contexto: las primeras horas de val/test toman L horas PASADAS del tramo anterior
    for cut, c in ((70, ORANGE), (85, RED)):
        ax.annotate("", xy=(cut + 2.2, 0.0), xytext=(cut - 3.5, 0.0),
                    arrowprops=dict(arrowstyle="->", color=c, lw=1.6))
        ax.barh(0.0, 3.5, left=cut - 3.5, color=c, alpha=0.25, height=0.2, hatch="///", edgecolor=c)
        ax.text(cut - 4.0, 0.0, "contexto de L h\n(solo pasado)", fontsize=7.5, color=c, ha="right", va="center")
        ax.axvline(cut, color=INK, ls="--", lw=1)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.3, 1.45)
    ax.set_yticks([])
    ax.set_xticks([0, 70, 85, 100])
    ax.set_xticklabels(["2011-01-01", "2012-05-27", "2012-09-12", "2012-12-31"], fontsize=8)
    ax.set_title("Validación por bloques cronológicos (sin barajar): el futuro nunca entra al ajuste", fontsize=10)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def fig_loss_grid(histories: Dict[str, Dict[str, List[float]]], grid: pd.DataFrame, selected: str):
    """Curvas de pérdida (MSE escalado) por época para las 9 configuraciones (3×3)."""
    fig, axes = plt.subplots(3, 3, figsize=(11, 7.2), sharex=True)
    ymin = min(min(h["val_loss"] + h["train_loss"]) for h in histories.values())
    ymax = max(max(h["val_loss"][1:] + h["train_loss"][1:]) for h in histories.values())
    for ax, (_, r) in zip(axes.ravel(), grid.iterrows()):
        h = histories[r["ID"]]
        ep = np.arange(1, len(h["train_loss"]) + 1)
        ax.plot(ep, h["train_loss"], color=BLUE, lw=1.6, label="Entrenamiento")
        ax.plot(ep, h["val_loss"], color=ORANGE, lw=1.6, label="Validación")
        ax.axvline(r["Mejor época"], color=GREY, ls="--", lw=1)
        ax.set_yscale("log")
        ax.set_ylim(ymin * 0.8, ymax * 1.3)
        sel = r["ID"] == selected
        ax.set_title(f"{r['ID']} · L={r['Ventana L']}, H={r['Hidden H']}" + ("  ★" if sel else ""),
                     fontsize=9, fontweight="bold" if sel else "normal", color=GREEN if sel else INK)
        if sel:
            for s in ax.spines.values():
                s.set_edgecolor(GREEN)
                s.set_linewidth(2)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.tick_params(labelsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("Época", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("MSE (escala [0,1])", fontsize=8)
    axes[0, 0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Pérdida MSE por época (línea gris = mejor época en validación; ★ = configuración elegida)",
                 fontsize=10)
    fig.tight_layout()
    return fig


def fig_loss_single(histories, ids: Iterable[str], colors: Optional[List[str]] = None):
    """Curvas de entrenamiento (punteadas) y validación (continuas) de varias configuraciones."""
    palette = colors or [BLUE, ORANGE, GREEN, RED, "#6A4C93", "#1B998B", "#E4572E", "#4D4D4D", "#B8860B"]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for i, cid in enumerate(ids):
        h = histories[cid]
        ep = np.arange(1, len(h["train_loss"]) + 1)
        c = palette[i % len(palette)]
        ax.plot(ep, h["train_loss"], color=c, ls=":", lw=1.4)
        ax.plot(ep, h["val_loss"], color=c, lw=2, label=f"{cid}")
    ax.set_yscale("log")
    ax.set_xlabel("Época")
    ax.set_ylabel("MSE (escala [0,1])")
    ax.set_title("Pérdida por época · continua = validación, punteada = entrenamiento")
    ax.legend(ncol=3, fontsize=8)
    _style(ax)
    fig.tight_layout()
    return fig


def fig_grid_heatmap(grid: pd.DataFrame, metric: str = "MAE val", selected: Optional[str] = None):
    """Mapa 3×3 (L × H) de la métrica de validación."""
    piv = grid.pivot(index="Ventana L", columns="Hidden H", values=metric)
    ids = grid.pivot(index="Ventana L", columns="Hidden H", values="ID")
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    im = ax.imshow(piv.to_numpy(), cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels([f"H = {h}" for h in piv.columns], fontsize=9)
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels([f"L = {l} h" for l in piv.index], fontsize=9)
    vals = piv.to_numpy()
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            cid = ids.iloc[i, j]
            star = " ★" if cid == selected else ""
            col = "white" if vals[i, j] > np.nanmin(vals) + 0.6 * (np.nanmax(vals) - np.nanmin(vals)) else INK
            ax.text(j, i, f"{cid}{star}\n{vals[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                    fontweight="bold", color=col)
    fig.colorbar(im, ax=ax, label=metric)
    ax.set_title(f"{metric} por configuración (menor es mejor)", fontsize=10)
    fig.tight_layout()
    return fig


def fig_complexity(grid: pd.DataFrame, selected: Optional[str] = None):
    """Compromiso error–complejidad: parámetros vs. MAE val (tamaño = tiempo)."""
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    colors = {12: BLUE, 24: ORANGE, 48: GREEN}
    t = grid["Tiempo (s)"].to_numpy(float)
    sizes = 60 + 340 * (t - t.min()) / max(1e-9, t.max() - t.min())
    for L, c in colors.items():
        m = grid["Ventana L"] == L
        ax.scatter(grid.loc[m, "Parámetros"], grid.loc[m, "MAE val"], s=sizes[m.to_numpy()], color=c,
                   alpha=0.75, edgecolor="white", label=f"L = {L} h")
    for _, r in grid.iterrows():
        ax.annotate(r["ID"] + (" ★" if r["ID"] == selected else ""), (r["Parámetros"], r["MAE val"]),
                    xytext=(6, 4), textcoords="offset points", fontsize=8,
                    fontweight="bold" if r["ID"] == selected else "normal")
    ax.set_xscale("log")
    ax.set_xlabel("Parámetros entrenables (escala log)")
    ax.set_ylabel("MAE validación (bicicletas/h)")
    ax.set_title("Error vs. complejidad (tamaño del punto = tiempo de entrenamiento)", fontsize=10)
    ax.legend(fontsize=8)
    _style(ax)
    fig.tight_layout()
    return fig


def fig_forecast(ts: pd.Series, y_true: np.ndarray, preds: Dict[str, np.ndarray], title: str,
                 colors: Optional[Dict[str, str]] = None, figsize=(11, 3.8)):
    """Demanda real frente a pronósticos en un tramo temporal."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(ts, y_true, color=INK, lw=2.2, label="Real cnt(t+1)")
    default = [GREEN, ORANGE, BLUE, RED]
    styles = ["-", "--", "-.", ":"]
    for i, (name, p) in enumerate(preds.items()):
        c = (colors or {}).get(name, default[i % len(default)])
        ax.plot(ts, p, color=c, lw=1.6 if i == 0 else 1.2, ls=styles[i % len(styles)], alpha=0.9, label=name)
    ax.set_ylabel("Alquileres por hora")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b\n%H:%M"))
    _style(ax)
    fig.tight_layout()
    return fig


def fig_test_bars(metrics: pd.DataFrame):
    """Barras de MAE y RMSE en prueba para RNN y baselines."""
    short = {PERSIST: "A. Persistencia", SEASONAL: "B. Prom. estacional k=7",
             SEASONAL_1: "B0. Día anterior", RIDGE: "C. Ridge rezagos"}
    labels = [short.get(m, "RNN seleccionada") for m in metrics["Modelo"]]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    b1 = ax.bar(x - 0.2, metrics["MAE"], 0.4, color=BLUE, label="MAE")
    b2 = ax.bar(x + 0.2, metrics["RMSE"], 0.4, color=ORANGE, label="RMSE")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{b.get_height():.1f}",
                    ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Bicicletas por hora")
    ax.set_title("Error en el conjunto de prueba (2 607 h, evaluación única)", fontsize=10)
    ax.legend(fontsize=8)
    _style(ax)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------------ gráficos explicativos (app)
PURPLE = "#6A4C93"
MODEL_COLORS = {"RNN": GREEN, "Persistencia": ORANGE, "Estacional": BLUE, "Día anterior": "#7FA7D9",
                "Ridge": PURPLE}


def _short_baseline(name: str) -> str:
    if name.startswith("A."):
        return "A · Persistencia"
    if name.startswith("B0."):
        return "B0 · Misma hora día anterior"
    if name.startswith("B."):
        return "B · Promedio estacional (7 días)"
    if name.startswith("C."):
        return "C · Ridge sobre rezagos"
    return name


def _baseline_color(name: str) -> str:
    if name.startswith("A."):
        return ORANGE
    if name.startswith("B0."):
        return "#7FA7D9"
    if name.startswith("B."):
        return BLUE
    return PURPLE


def fig_window_example(df: pd.DataFrame, i_tr: int, L: int, j: int, after: int = 24):
    """Una muestra de validación: ventana de entrada (pasado) y objetivo t+1, junto al corte Train/Val."""
    tgt = i_tr + j
    lo, hi = max(0, tgt - L - 24), min(len(df) - 1, tgt + after)
    ts = df["timestamp"].iloc[lo:hi + 1]
    y = df["cnt"].iloc[lo:hi + 1]
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(ts, y, color=GREY, lw=1.3, zorder=1, label="Serie cnt")
    ymax = float(y.max()) * 1.15
    cut = df["timestamp"].iloc[i_tr]
    if ts.iloc[0] < cut:
        ax.axvspan(ts.iloc[0], cut, color=BLUE, alpha=0.07)
        ax.text(cut, ymax * 0.97, "← Entrenamiento  ", color=BLUE, fontsize=9, va="top", ha="right")
    ax.axvspan(max(cut, ts.iloc[0]), ts.iloc[-1], color=ORANGE, alpha=0.07)
    ax.axvline(cut, color=INK, ls="--", lw=1)
    w_ts = df["timestamp"].iloc[tgt - L:tgt]
    ax.axvspan(w_ts.iloc[0], w_ts.iloc[-1], color=GREEN, alpha=0.22, zorder=0)
    ax.plot(w_ts, df["cnt"].iloc[tgt - L:tgt], color=GREEN, lw=2.4, label=f"Entrada: ventana de L = {L} h (≤ t)")
    ax.scatter([df["timestamp"].iloc[tgt]], [df["cnt"].iloc[tgt]], s=120, color=RED, zorder=5, marker="*",
               label="Objetivo: cnt(t+1)")
    ax.set_ylim(0, ymax)
    if ts.iloc[0] <= cut <= ts.iloc[-1]:
        ax.text(cut, ymax * 0.97, "  Validación →", color=ORANGE, fontsize=9, va="top")
    ax.set_ylabel("Alquileres por hora")
    ax.set_title("La red solo ve el tramo verde (pasado) para predecir la estrella roja (hora siguiente)", fontsize=10)
    ax.legend(loc="upper left", fontsize=8.5, bbox_to_anchor=(0, 0.88), framealpha=0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b\n%H:%M"))
    _style(ax)
    fig.tight_layout()
    return fig


def fig_model_ranking(grid: pd.DataFrame, baselines: pd.DataFrame, selected: str, metric: str = "MAE"):
    """Barras horizontales: las 9 RNN y los baselines ordenados por error de validación."""
    rows = [(f"RNN {r['ID']} (L={r['Ventana L']}, H={r['Hidden H']})", r[f"{metric} val"],
             GREEN if r["ID"] == selected else "#8CC5A1") for _, r in grid.iterrows()]
    rows += [(_short_baseline(m), v, _baseline_color(m)) for m, v in baselines[metric].items()]
    rows.sort(key=lambda x: x[1], reverse=True)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], color=[r[2] for r in rows], height=0.7)
    for i, (lab, v, _) in enumerate(rows):
        ax.text(v + 0.8, i, f"{v:.1f}", va="center", fontsize=9,
                fontweight="bold" if lab.startswith(f"RNN {selected} ") else "normal")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] + ("  ★" if r[0].startswith(f"RNN {selected} ") else "") for r in rows], fontsize=9)
    ax.set_xlabel(f"{metric} en validación (bicicletas/hora) · menor es mejor")
    ax.set_title(f"Ranking de modelos por {metric} en validación (2 607 h)", fontsize=11)
    ax.set_xlim(0, max(r[1] for r in rows) * 1.12)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def fig_effects(grid: pd.DataFrame, metric: str = "MAE val"):
    """Efecto de L (eje x) para cada H (líneas)."""
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for H, c in zip(sorted(grid["Hidden H"].unique()), (BLUE, ORANGE, GREEN)):
        sub = grid[grid["Hidden H"] == H].sort_values("Ventana L")
        ax.plot(sub["Ventana L"], sub[metric], marker="o", color=c, lw=2, label=f"H = {H}")
        for _, r in sub.iterrows():
            ax.annotate(r["ID"], (r["Ventana L"], r[metric]), xytext=(4, 4), textcoords="offset points",
                        fontsize=7.5, color=c)
    ax.set_xticks(sorted(grid["Ventana L"].unique()))
    ax.set_xlabel("Ventana L (horas de historia)")
    ax.set_ylabel(f"{metric} (bicicletas/h)")
    ax.set_title("¿Cuánta historia y cuántas neuronas convienen?", fontsize=10)
    ax.legend(fontsize=8, title="Unidades ocultas", title_fontsize=8)
    _style(ax)
    fig.tight_layout()
    return fig


def fig_loss_focus(history: Dict[str, List[float]], cid: str, best_epoch: int, patience: int = 8):
    """Curva de una configuración con anotaciones didácticas (mejor época, early stopping, brecha)."""
    tr, va = history["train_loss"], history["val_loss"]
    ep = np.arange(1, len(tr) + 1)
    fig, ax = plt.subplots(figsize=(9.5, 3.9))
    ax.plot(ep, tr, color=BLUE, lw=2.2, label="Entrenamiento (datos que la red ve)")
    ax.plot(ep, va, color=ORANGE, lw=2.2, label="Validación (datos nuevos)")
    ax.fill_between(ep, tr, va, color=ORANGE, alpha=0.08)
    ax.axvline(best_epoch, color=GREEN, lw=1.6, ls="--")
    ax.scatter([best_epoch], [va[best_epoch - 1]], color=GREEN, s=80, zorder=5)
    ax.annotate(f"Mejor época = {best_epoch}\nse guardan estos pesos", (best_epoch, va[best_epoch - 1]),
                xytext=(-130, 45), textcoords="offset points", fontsize=8.5, color=GREEN,
                arrowprops=dict(arrowstyle="->", color=GREEN))
    if len(ep) > best_epoch:
        ax.axvspan(best_epoch, ep[-1], color=GREY, alpha=0.12)
        ax.text((best_epoch + ep[-1]) / 2, min(tr) * 0.82, "sin mejora:\nse detiene", ha="center",
                va="top", fontsize=8, color=INK)
    ax.text(ep[len(ep) // 3], np.sqrt(tr[len(ep) // 3] * va[len(ep) // 3]), "brecha val/train", fontsize=8,
            color=ORANGE, ha="center", va="center")
    ax.set_yscale("log")
    ax.set_ylim(min(tr) * 0.55, max(va) * 1.3)
    ax.set_xlabel("Época (pasada completa por los datos de entrenamiento)")
    ax.set_ylabel("Pérdida MSE (escala log)")
    ax.set_title(f"Cómo aprendió {cid}: ambas curvas bajan y la validación se estabiliza", fontsize=10)
    ax.legend(fontsize=8.5, loc="upper right")
    _style(ax)
    fig.tight_layout()
    return fig


def fig_gain_bars(model_mae: float, baselines: Dict[str, float], title: str):
    """Reducción porcentual del MAE de la RNN frente a cada baseline."""
    names = list(baselines)
    gains = [(1 - model_mae / v) * 100 for v in baselines.values()]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    bars = ax.barh(names, gains, color=[_baseline_color(n) for n in names], height=0.6)
    for b, gv in zip(bars, gains):
        ax.text(gv + 1, b.get_y() + b.get_height() / 2, f"−{gv:.0f} %", va="center", fontsize=9, fontweight="bold")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([_short_baseline(n) for n in names], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Reducción del MAE lograda por la RNN (%)")
    ax.set_title(title, fontsize=10)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def fig_test_ranking(metrics: pd.DataFrame):
    """MAE en prueba por modelo (barras horizontales) con mejora vs. persistencia."""
    m = metrics.sort_values("MAE", ascending=False)
    labels = ["RNN seleccionada" if n.startswith("RNN") else _short_baseline(n) for n in m["Modelo"]]
    colors = [GREEN if n.startswith("RNN") else _baseline_color(n) for n in m["Modelo"]]
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    y = np.arange(len(m))
    ax.barh(y, m["MAE"], color=colors, height=0.65)
    for i, (v, imp) in enumerate(zip(m["MAE"], m["Mejora MAE vs. persistencia (%)"])):
        if abs(imp) < 1e-9:
            tag = "referencia"
        elif imp > 0:
            tag = f"{imp:.0f} % menos error que persistencia"
        else:
            tag = f"{-imp:.0f} % más error que persistencia"
        ax.text(v + 1, i, f"{v:.1f}  ({tag})", va="center", fontsize=8.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, m["MAE"].max() * 1.45)
    ax.set_xlabel("MAE en prueba (bicicletas/hora) · menor es mejor")
    ax.set_title("Prueba final (2 607 h nunca vistas): error de cada modelo", fontsize=10)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def fig_real_vs_pred(tp: pd.DataFrame, cols=(("rnn", "RNN seleccionada", GREEN),
                                             ("persistencia", "A · Persistencia", ORANGE))):
    """Dispersión real vs. predicho: cuanto más cerca de la diagonal, mejor."""
    fig, axes = plt.subplots(1, len(cols), figsize=(9.5, 3.9), sharex=True, sharey=True)
    lim = float(max(tp["y_true"].max(), *(tp[c].max() for c, _, _ in cols))) * 1.03
    for ax, (c, lab, colr) in zip(np.atleast_1d(axes), cols):
        ax.scatter(tp["y_true"], tp[c], s=6, alpha=0.25, color=colr)
        ax.plot([0, lim], [0, lim], color=INK, lw=1, ls="--")
        ax.set_title(f"{lab} · MAE = {mae(tp['y_true'], tp[c]):.1f}", fontsize=10)
        ax.set_xlabel("Demanda real")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        _style(ax)
    np.atleast_1d(axes)[0].set_ylabel("Demanda pronosticada")
    fig.suptitle("Real vs. pronosticado en prueba (línea punteada = pronóstico perfecto)", fontsize=10)
    fig.tight_layout()
    return fig


def best_week_start(ts: pd.Series, offset_days: int = 14) -> int:
    """Índice del primer lunes 00:00 tras `offset_days` días (tramo semanal legible)."""
    ts = pd.to_datetime(ts).reset_index(drop=True)
    cand = ts[(ts.dt.dayofweek == 0) & (ts.dt.hour == 0)]
    cand = cand[cand >= ts.iloc[0] + pd.Timedelta(days=offset_days)]
    return int(cand.index[0]) if len(cand) else 0


def save_report_figures(results_dir: Path = RESULTS_DIR) -> Dict[str, Path]:
    """Exporta las figuras del informe (Figuras 5, 6 y 7 y apoyos) a results/figures/."""
    res = load_results(results_dir)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sel = res["selection"]
    grid = res["grid"]
    paths: Dict[str, Path] = {}

    def _save(fig, name):
        p = FIG_DIR / name
        fig.savefig(p, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths[name] = p

    _save(fig_protocol(), "protocolo_validacion.png")
    _save(fig_loss_grid(res["histories"], grid, sel["ID"]), "fig5_curvas_perdida.png")
    _save(fig_grid_heatmap(grid, "MAE val", sel["ID"]), "rejilla_mae_val.png")
    _save(fig_complexity(grid, sel["ID"]), "error_vs_complejidad.png")

    vp = res["val_predictions"]
    s = best_week_start(vp["timestamp"])
    w = slice(s, s + 168)
    _save(fig_forecast(vp["timestamp"].iloc[w], vp["y_true"].iloc[w].to_numpy(),
                       {f"RNN {sel['ID']}": vp[sel["ID"]].iloc[w].to_numpy(),
                        "A. Persistencia": vp["persistencia"].iloc[w].to_numpy(),
                        "B. Promedio estacional (k=7)": vp["promedio_estacional"].iloc[w].to_numpy()},
                       f"Validación · semana desde {vp['timestamp'].iloc[s]:%Y-%m-%d}: real vs. pronóstico"),
          "fig6_pronostico_validacion.png")

    if res["test_predictions"] is not None:
        tp = res["test_predictions"]
        s = best_week_start(tp["timestamp"])
        w = slice(s, s + 168)
        _save(fig_forecast(tp["timestamp"].iloc[w], tp["y_true"].iloc[w].to_numpy(),
                           {f"RNN {sel['ID']}": tp["rnn"].iloc[w].to_numpy(),
                            "A. Persistencia": tp["persistencia"].iloc[w].to_numpy(),
                            "B. Promedio estacional (k=7)": tp["promedio_estacional"].iloc[w].to_numpy()},
                           f"Prueba · semana desde {tp['timestamp'].iloc[s]:%Y-%m-%d}: real vs. pronóstico"),
              "fig7_pronostico_prueba.png")
        _save(fig_test_bars(res["test_metrics"]), "test_mae_rmse.png")
    return paths


if __name__ == "__main__":
    evaluate_on_test(verbose=True)
    for name, p in save_report_figures().items():
        print(f"Figura guardada: {p}")
