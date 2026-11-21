"""Ejecución experimental de la rejilla 3×3 y selección de configuración (Parte 4).

Cubre las secciones 4.7, 5.1, 5.2 y 5.3 del informe:
- Entrena las 9 configuraciones C1–C9 (L ∈ {12, 24, 48} × H ∈ {16, 32, 64}) con
  `train.train_rnn`, usando la misma partición, escalador, semilla y TrainConfig.
- Validación por bloque temporal: el modelo se ajusta en Train (70 %) y se evalúa en las
  2 607 horas siguientes (Val, 15 %). El conjunto de prueba NO se toca aquí.
- Selecciona la configuración con menor MAE de validación (desempate por RMSE y luego
  por menor número de parámetros).
- Guarda todo en `results/` para que la app (pestañas 4 y 5) solo lea resultados.

Uso:
    python experiments.py            # rejilla completa (≈ minutos en CPU)
    python evaluate.py               # evaluación única en prueba de la configuración elegida
"""
from __future__ import annotations

import json
import platform
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

from baselines import evaluate_baselines
from config import (
    DEFAULT_SEASONAL_DAYS,
    FEATURES,
    HIDDEN_SIZES,
    RESULTS_DIR,
    SEED,
    TRAIN_FRAC,
    VAL_FRAC,
    WINDOW_SIZES,
)
from data import load_hour
from features import add_cyclic_features
from train import TrainConfig, train_rnn

MODELS_DIR = RESULTS_DIR / "models"

# Experimentos del plan (Tabla 11): C = memoria corta; D = dependencias más largas
EXP_C_IDS = ["C1"]
EXP_D_IDS = ["C5", "C6", "C8", "C9"]


def grid_configs(window_sizes=WINDOW_SIZES, hidden_sizes=HIDDEN_SIZES) -> List[Dict[str, Any]]:
    """Lista ordenada C1..C9 con (L, H), igual que la Tabla 9 de la Parte 3."""
    out, idx = [], 1
    for L in window_sizes:
        for H in hidden_sizes:
            out.append({"ID": f"C{idx}", "L": L, "H": H})
            idx += 1
    return out


def select_best(grid_df: pd.DataFrame) -> pd.Series:
    """Criterio de selección (sección 5.3): menor MAE val → menor RMSE val → menos parámetros."""
    ranked = grid_df.sort_values(["MAE val", "RMSE val", "Parámetros"], ascending=True)
    return ranked.iloc[0]


def environment_info(device: str) -> Dict[str, str]:
    """Versiones de bibliotecas y entorno (Tabla 10 del informe)."""
    import matplotlib
    import streamlit

    try:
        import sklearn  # noqa: F401

        sk_version = sklearn.__version__
    except ImportError:
        sk_version = "no utilizado"

    return {
        "Python": platform.python_version(),
        "PyTorch": torch.__version__,
        "NumPy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sk_version,
        "Matplotlib": matplotlib.__version__,
        "Streamlit": streamlit.__version__,
        "Semilla": str(SEED),
        "Dispositivo": device.upper(),
        "Procesador": platform.processor() or platform.machine(),
        "Sistema operativo": f"{platform.system()} {platform.release()}",
        "Hilos CPU (torch)": str(torch.get_num_threads()),
    }


def run_grid(verbose: bool = True) -> Dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = add_cyclic_features(load_hour())
    cfg = TrainConfig()
    configs = grid_configs()

    rows: List[Dict[str, Any]] = []
    histories: Dict[str, Dict[str, List[float]]] = {}
    val_preds: Dict[str, np.ndarray] = {}
    val_true = val_ts = None

    for c in configs:
        if verbose:
            print(f"\n=== {c['ID']} · L = {c['L']} h, H = {c['H']} ===")
        res = train_rnn(L=c["L"], hidden_size=c["H"], df=df, config=cfg, verbose=verbose)

        rows.append({
            "ID": c["ID"],
            "Ventana L": c["L"],
            "Hidden H": c["H"],
            "Capas": res["num_layers"],
            "Parámetros": res["n_params"],
            "Tiempo (s)": round(res["train_time_sec"], 2),
            "Épocas": res["epochs_run"],
            "Mejor época": res["best_epoch"],
            "MSE val (escalado)": res["best_val_mse"],
            "MAE val": res["val_metrics"]["mae"],
            "RMSE val": res["val_metrics"]["rmse"],
            "MASE val": res["val_metrics"]["mase"],
        })
        histories[c["ID"]] = res["history"]
        val_preds[c["ID"]] = res["val_preds"]
        val_true, val_ts = res["val_true"], res["val_timestamps"]

        torch.save(
            {
                "state_dict": res["best_state_dict"],
                "L": c["L"],
                "hidden_size": c["H"],
                "num_layers": res["num_layers"],
                "input_size": len(FEATURES),
                "features": FEATURES,
            },
            MODELS_DIR / f"{c['ID']}.pt",
        )
        if verbose:
            r = rows[-1]
            print(f"→ {c['ID']}: MAE val = {r['MAE val']:.2f} | RMSE val = {r['RMSE val']:.2f} | "
                  f"{r['Tiempo (s)']:.1f} s | mejor época {r['Mejor época']}/{r['Épocas']}")

    grid_df = pd.DataFrame(rows)
    best = select_best(grid_df)
    grid_df["Seleccionada"] = grid_df["ID"] == best["ID"]

    # Baselines sobre exactamente las mismas 2 607 horas de validación
    base_table, base_preds, y_val_b, _ = evaluate_baselines(
        df, split="val", seasonal_days=DEFAULT_SEASONAL_DAYS, include_lag_reg=True,
        train_frac=TRAIN_FRAC, val_frac=VAL_FRAC,
    )
    assert np.allclose(y_val_b, val_true), "Baselines y RNN deben compartir las mismas horas de validación"

    # ---- Persistencia de artefactos
    grid_df.to_csv(RESULTS_DIR / "grid_results.csv", index=False)
    base_table.to_csv(RESULTS_DIR / "baselines_val.csv", index=False)
    with open(RESULTS_DIR / "histories.json", "w", encoding="utf-8") as f:
        json.dump(histories, f)

    val_out = pd.DataFrame({"timestamp": val_ts, "y_true": val_true})
    for cid, p in val_preds.items():
        val_out[cid] = p
    val_out["persistencia"] = base_preds["A. Persistencia: cnt(t)"]
    val_out["promedio_estacional"] = base_preds[f"B. Promedio estacional (k={DEFAULT_SEASONAL_DAYS} días)"]
    val_out["ridge"] = base_preds["C. Regresión Ridge sobre rezagos (opcional)"]
    val_out.to_csv(RESULTS_DIR / "val_predictions.csv", index=False)

    selection = {
        "ID": best["ID"],
        "L": int(best["Ventana L"]),
        "H": int(best["Hidden H"]),
        "num_layers": int(best["Capas"]),
        "n_params": int(best["Parámetros"]),
        "train_time_sec": float(best["Tiempo (s)"]),
        "best_epoch": int(best["Mejor época"]),
        "epochs_run": int(best["Épocas"]),
        "val_mae": float(best["MAE val"]),
        "val_rmse": float(best["RMSE val"]),
        "val_mase": float(best["MASE val"]),
        "criterio": "Menor MAE en validación; desempate por RMSE y luego por menos parámetros",
    }
    with open(RESULTS_DIR / "selection.json", "w", encoding="utf-8") as f:
        json.dump(selection, f, indent=2, ensure_ascii=False)

    env = environment_info(cfg.resolved_device())
    env["Comando"] = " ".join([sys.executable.split("\\")[-1]] + sys.argv)
    with open(RESULTS_DIR / "environment.json", "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2, ensure_ascii=False)
    with open(RESULTS_DIR / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    if verbose:
        print("\n" + grid_df.to_string(index=False))
        print(f"\nConfiguración seleccionada: {best['ID']} (L = {best['Ventana L']}, H = {best['Hidden H']})")
        print(f"Resultados guardados en {RESULTS_DIR}")
    return {"grid": grid_df, "selection": selection}


def verify_reproducibility(cid: str = "C1") -> Dict[str, Any]:
    """Reentrena una configuración en una ejecución nueva y compara con lo guardado en grid_results.csv."""
    grid_df = pd.read_csv(RESULTS_DIR / "grid_results.csv")
    ref = grid_df[grid_df["ID"] == cid].iloc[0]
    res = train_rnn(L=int(ref["Ventana L"]), hidden_size=int(ref["Hidden H"]),
                    df=add_cyclic_features(load_hour()), config=TrainConfig())
    out = {
        "ID": cid,
        "MAE grid": float(ref["MAE val"]), "MAE reentrenado": res["val_metrics"]["mae"],
        "RMSE grid": float(ref["RMSE val"]), "RMSE reentrenado": res["val_metrics"]["rmse"],
        "Mejor época grid": int(ref["Mejor época"]), "Mejor época reentrenado": res["best_epoch"],
    }
    out["Idéntico"] = bool(np.isclose(out["MAE grid"], out["MAE reentrenado"], atol=1e-6)
                           and out["Mejor época grid"] == out["Mejor época reentrenado"])
    with open(RESULTS_DIR / "reproducibility.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return out


if __name__ == "__main__":
    run_grid(verbose=True)
    print("\nVerificación de reproducibilidad:", verify_reproducibility("C1"))
