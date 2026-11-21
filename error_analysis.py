"""Análisis detallado de errores para el pronóstico horario de alquileres (Parte 5).

Cubre la sección 5.5 del informe:
- Desglose de MAE y RMSE por hora del día (0 a 23).
- Comparación en días laborales vs. fines de semana y feriados.
- Identificación y evaluación de días de mayor demanda y picos horarios.
- Impacto de condiciones meteorológicas adversas sobre el error predictivo.
"""
from __future__ import annotations

from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import RESULTS_DIR
from metrics import mae, rmse

BLUE, ORANGE, GREEN, RED, INK, GREY = "#2B579A", "#C8581A", "#2E7D4F", "#B5251D", "#1F2A44", "#8A94A6"

WEATHER_NAMES = {
    1: "Despejado / Pocas nubes",
    2: "Niebla / Nublado",
    3: "Lluvia ligera / Nieve",
    4: "Tormenta / Lluvia fuerte",
}


def load_test_predictions(csv_path=None) -> pd.DataFrame:
    """Carga el dataset de predicciones de prueba con formato temporal."""
    path = csv_path or (RESULTS_DIR / "test_predictions.csv")
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df["error_rnn"] = np.abs(df["y_true"] - df["rnn"])
    df["error_persist"] = np.abs(df["y_true"] - df["persistencia"])
    df["error_seasonal"] = np.abs(df["y_true"] - df["promedio_estacional"])
    return df


def compute_hourly_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula MAE, RMSE y demanda media para cada hora del día (0–23)."""
    rows = []
    for hr in range(24):
        sub = df[df["hr"] == hr]
        if len(sub) == 0:
            continue
        y = sub["y_true"].to_numpy()
        p_rnn = sub["rnn"].to_numpy()
        p_per = sub["persistencia"].to_numpy()
        p_sea = sub["promedio_estacional"].to_numpy()

        rows.append({
            "hr": hr,
            "muestras": len(sub),
            "demanda_media": float(y.mean()),
            "mae_rnn": mae(y, p_rnn),
            "rmse_rnn": rmse(y, p_rnn),
            "mae_persist": mae(y, p_per),
            "rmse_persist": rmse(y, p_per),
            "mae_seasonal": mae(y, p_sea),
            "rmse_seasonal": rmse(y, p_sea),
            "mejora_vs_persist_%": (1.0 - mae(y, p_rnn) / max(1e-6, mae(y, p_per))) * 100.0,
        })
    return pd.DataFrame(rows)


def compute_workingday_hourly_metrics(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Separa el error horario entre días laborables y fines de semana/feriados."""
    laboral = compute_hourly_metrics(df[df["workingday"] == 1])
    no_laboral = compute_hourly_metrics(df[df["workingday"] == 0])
    return laboral, no_laboral


def compute_peak_days(df: pd.DataFrame, top_n: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Identifica los días de mayor demanda diaria acumulada en el conjunto de prueba."""
    daily = df.groupby("date").agg(
        demanda_total=("y_true", "sum"),
        demanda_media=("y_true", "mean"),
        pico_max=("y_true", "max"),
        mae_rnn=("error_rnn", "mean"),
        mae_persist=("error_persist", "mean"),
        mae_seasonal=("error_seasonal", "mean"),
        workingday=("workingday", "first"),
        weathersit_moda=("weathersit", lambda x: x.mode().iloc[0] if len(x) else 1),
    ).reset_index()

    daily["mejora_vs_persist_%"] = (1.0 - daily["mae_rnn"] / daily["mae_persist"]) * 100.0
    daily["dia_semana"] = pd.to_datetime(daily["date"]).dt.day_name()
    
    top_days = daily.sort_values("demanda_total", ascending=False).head(top_n).reset_index(drop=True)
    summary_comparison = pd.DataFrame([
        {
            "Grupo": f"Top {top_n} días mayor demanda",
            "Demanda media horaria": top_days["demanda_media"].mean(),
            "MAE RNN": top_days["mae_rnn"].mean(),
            "MAE Persistencia": top_days["mae_persist"].mean(),
            "Mejora RNN (%)": (1.0 - top_days["mae_rnn"].mean() / top_days["mae_persist"].mean()) * 100.0,
        },
        {
            "Grupo": "Resto de días de prueba",
            "Demanda media horaria": daily.drop(top_days.index)["demanda_media"].mean(),
            "MAE RNN": daily.drop(top_days.index)["mae_rnn"].mean(),
            "MAE Persistencia": daily.drop(top_days.index)["mae_persist"].mean(),
            "Mejora RNN (%)": (1.0 - daily.drop(top_days.index)["mae_rnn"].mean() / daily.drop(top_days.index)["mae_persist"].mean()) * 100.0,
        },
        {
            "Grupo": "Promedio total prueba",
            "Demanda media horaria": daily["demanda_media"].mean(),
            "MAE RNN": daily["mae_rnn"].mean(),
            "MAE Persistencia": daily["mae_persist"].mean(),
            "Mejora RNN (%)": (1.0 - daily["mae_rnn"].mean() / daily["mae_persist"].mean()) * 100.0,
        }
    ])
    return top_days, summary_comparison


def compute_weather_impact(df: pd.DataFrame) -> pd.DataFrame:
    """Analiza cómo afecta el estado del tiempo (`weathersit`) a los errores de pronóstico."""
    rows = []
    for sit, name in WEATHER_NAMES.items():
        sub = df[df["weathersit"] == sit]
        if len(sub) == 0:
            continue
        y = sub["y_true"].to_numpy()
        p_rnn = sub["rnn"].to_numpy()
        p_per = sub["persistencia"].to_numpy()
        rows.append({
            "Código": sit,
            "Clima": name,
            "Horas": len(sub),
            "Demanda media": float(y.mean()),
            "MAE RNN": mae(y, p_rnn),
            "RMSE RNN": rmse(y, p_rnn),
            "MAE Persistencia": mae(y, p_per),
            "Mejora vs Persist (%)": (1.0 - mae(y, p_rnn) / max(1e-6, mae(y, p_per))) * 100.0,
        })
    return pd.DataFrame(rows)


# ======================================================================== Gráficos
def fig_hourly_error(hourly: pd.DataFrame) -> plt.Figure:
    """Curva de error MAE por hora del día para RNN y baselines."""
    fig, ax1 = plt.subplots(figsize=(10, 4.2))

    ax1.plot(hourly["hr"], hourly["mae_rnn"], marker="o", color=GREEN, lw=2.4, label="MAE RNN seleccionada")
    ax1.plot(hourly["hr"], hourly["mae_persist"], marker="s", color=ORANGE, lw=1.8, ls="--", label="MAE Persistencia")
    ax1.plot(hourly["hr"], hourly["mae_seasonal"], marker="^", color=BLUE, lw=1.8, ls=":", label="MAE Promedio estacional (k=7)")

    # Sombreado de horas pico habituales (08:00 y 17:00-18:00)
    ax1.axvspan(7.5, 8.5, color=GREY, alpha=0.18, label="Picos laborales (08:00 y 17-18:00)")
    ax1.axvspan(16.5, 18.5, color=GREY, alpha=0.18)

    ax1.set_xlabel("Hora del día (0 a 23)", fontsize=9.5)
    ax1.set_ylabel("MAE (bicicletas/hora)", fontsize=9.5)
    ax1.set_xticks(range(0, 24))
    ax1.set_title("Evolución del Error Absoluto Medio (MAE) a lo largo del ciclo diario", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc="upper left", fontsize=8.5)

    # Eje secundario con la demanda media real
    ax2 = ax1.twinx()
    ax2.plot(hourly["hr"], hourly["demanda_media"], color=INK, lw=1.2, alpha=0.35, ls="-.", label="Demanda media real")
    ax2.set_ylabel("Demanda media (bicicletas/h)", color=INK, fontsize=8.5, alpha=0.7)
    ax2.tick_params(axis="y", labelcolor=INK, labelsize=8)

    fig.tight_layout()
    return fig


def fig_hourly_by_workingday(lab: pd.DataFrame, no_lab: pd.DataFrame) -> plt.Figure:
    """Comparativa en dos subgráficos de días laborables vs. fines de semana / feriados."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.0), sharey=True)

    # Días laborables
    ax1.plot(lab["hr"], lab["mae_rnn"], marker="o", color=GREEN, lw=2.2, label="RNN")
    ax1.plot(lab["hr"], lab["mae_persist"], marker="s", color=ORANGE, lw=1.8, ls="--", label="Persistencia")
    ax1.plot(lab["hr"], lab["mae_seasonal"], marker="^", color=BLUE, lw=1.8, ls=":", label="Prom. estacional")
    ax1.set_title("Días Laborables (Picos en desplazamientos)", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Hora")
    ax1.set_ylabel("MAE (bicicletas/hora)")
    ax1.set_xticks(range(0, 24, 2))
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(fontsize=8)

    # Fines de semana / feriados
    ax2.plot(no_lab["hr"], no_lab["mae_rnn"], marker="o", color=GREEN, lw=2.2, label="RNN")
    ax2.plot(no_lab["hr"], no_lab["mae_persist"], marker="s", color=ORANGE, lw=1.8, ls="--", label="Persistencia")
    ax2.plot(no_lab["hr"], no_lab["mae_seasonal"], marker="^", color=BLUE, lw=1.8, ls=":", label="Prom. estacional")
    ax2.set_title("Fines de Semana / Feriados (Curva suave recreativa)", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Hora")
    ax2.set_xticks(range(0, 24, 2))
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    return fig


def fig_weather_error(weather_df: pd.DataFrame) -> plt.Figure:
    """Gráfico de barras de MAE por tipo de clima."""
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    x = np.arange(len(weather_df))
    w = 0.35

    ax.bar(x - w / 2, weather_df["MAE RNN"], w, label="RNN seleccionada", color=GREEN, alpha=0.9)
    ax.bar(x + w / 2, weather_df["MAE Persistencia"], w, label="Persistencia", color=ORANGE, alpha=0.9)

    for i, row in weather_df.iterrows():
        ax.text(i - w / 2, row["MAE RNN"] + 1.5, f"{row['MAE RNN']:.1f}", ha="center", fontsize=8.5, fontweight="bold")
        ax.text(i + w / 2, row["MAE Persistencia"] + 1.5, f"{row['MAE Persistencia']:.1f}", ha="center", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(weather_df["Clima"], rotation=12, ha="right", fontsize=8.5)
    ax.set_ylabel("MAE (bicicletas/hora)")
    ax.set_title("Sensibilidad del Error frente a las Condiciones Meteorológicas", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8.5)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    preds = load_test_predictions()
    print("Muestras de prueba cargadas:", len(preds))
    hourly = compute_hourly_metrics(preds)
    print("\nResumen error por hora (primeras 6 horas):")
    print(hourly.head(6)[["hr", "demanda_media", "mae_rnn", "mae_persist", "mejora_vs_persist_%"]])
    top, comp = compute_peak_days(preds, top_n=5)
    print("\nComparación picos vs resto:")
    print(comp)
