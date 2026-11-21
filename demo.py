"""Módulo de demostración interactiva en tiempo real y advertencias climáticas (Parte 5).

Cubre la sección 7 del informe y la funcionalidad del simulador en Streamlit:
- Selección de un instante histórico dentro del conjunto de prueba.
- Visualización de la trayectoria reciente de las últimas 24 o 48 horas.
- Pronóstico puntual de la siguiente hora inmediata (t+1) con la RNN vs. valor real.
- Sistema de detección de advertencias climáticas operativas (lluvia, nieve, temperaturas extremas, vientos).
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import RESULTS_DIR

BLUE, ORANGE, GREEN, RED, INK, GREY = "#2B579A", "#C8581A", "#2E7D4F", "#B5251D", "#1F2A44", "#8A94A6"


def check_weather_alerts(row: pd.Series) -> Dict[str, Any]:
    """Evalúa variables meteorológicas en t y genera advertencias operativas para t+1."""
    weathersit = int(row.get("weathersit", 1))
    temp_norm = float(row.get("temp", 0.5))
    hum_norm = float(row.get("hum", 0.5))
    wind_norm = float(row.get("windspeed", 0.2))

    # Desnormalización aproximada según metadatos UCI (t_min=-8°C, t_max=39°C, w_max=67 km/h)
    temp_c = temp_norm * 47.0 - 8.0
    wind_kmh = wind_norm * 67.0
    hum_pct = hum_norm * 100.0

    alerts = []
    level = "info"  # "info", "warning", "critical"

    # 1. Alertas por precipitación o severidad climática
    if weathersit == 4:
        alerts.append("⛈️ **Alerta Crítica:** Tormenta eléctrica o precipitación severa. Riesgo operativo alto.")
        level = "critical"
    elif weathersit == 3:
        alerts.append("🌧️ **Advertencia Severa:** Lluvia o nieve ligera/moderada. Se anticipa fuerte contracción de demanda.")
        if level != "critical":
            level = "warning"
    elif weathersit == 2:
        alerts.append("🌫️ **Aviso:** Nubosidad o niebla densa; visibilidad reducida.")

    # 2. Alertas por temperatura
    if temp_c < 2.0:
        alerts.append(f"🥶 **Baja Temperatura ({temp_c:.1f} °C):** Posible presencia de escarcha o hielo en calzadas.")
        if level != "critical":
            level = "warning"
    elif temp_c > 33.0:
        alerts.append(f"☀️ **Ola de Calor ({temp_c:.1f} °C):** Demanda puede reducirse en horas de máxima insolación.")

    # 3. Alertas por viento
    if wind_kmh > 32.0:
        alerts.append(f"💨 **Viento Fuerte ({wind_kmh:.1f} km/h):** Mayor dificultad de pedaleo; puede afectar rotación de bicicletas.")
        if level != "critical":
            level = "warning"

    if not alerts:
        alerts.append(f"🌤️ **Condiciones Favorables ({temp_c:.1f} °C, {wind_kmh:.1f} km/h, {hum_pct:.0f}% humedad):** Operación en rango óptimo.")

    return {
        "level": level,
        "messages": alerts,
        "temp_c": temp_c,
        "wind_kmh": wind_kmh,
        "hum_pct": hum_pct,
        "weathersit": weathersit,
    }


def get_demo_slice(
    df: pd.DataFrame,
    target_idx: int,
    window_hours: int = 24,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Extrae la ventana de historial [t - L + 1, t] y la fila objetivo t+1."""
    if target_idx < window_hours:
        target_idx = window_hours
    if target_idx >= len(df):
        target_idx = len(df) - 1

    history_df = df.iloc[target_idx - window_hours : target_idx].copy().reset_index(drop=True)
    target_row = df.iloc[target_idx].copy()
    return history_df, target_row


def fig_demo_timeline(
    history_df: pd.DataFrame,
    target_row: pd.Series,
    window_hours: int = 24,
) -> plt.Figure:
    """Genera la figura interactiva de la demo que muestra el pasado y el pronóstico de la siguiente hora."""
    fig, ax = plt.subplots(figsize=(11, 4.2))

    # Serie temporal observada en la ventana (≤ t)
    ax.plot(
        history_df["timestamp"],
        history_df["y_true"],
        color=INK,
        lw=2.2,
        marker="o",
        markersize=4,
        label=f"Historial observado (últimas {window_hours} h hasta t)",
    )

    # Predicción pasada de la RNN para contexto
    ax.plot(
        history_df["timestamp"],
        history_df["rnn"],
        color=GREEN,
        lw=1.6,
        ls="--",
        alpha=0.75,
        label="RNN (estimaciones pasadas)",
    )

    # Línea vertical divisoria entre presente (t) y futuro (t+1)
    t_stamp = history_df["timestamp"].iloc[-1]
    next_stamp = target_row["timestamp"]
    ax.axvline(t_stamp, color=GREY, ls=":", lw=1.8, label="Instante actual de decisión (t)")

    # Punto objetivo real cnt(t+1)
    y_true_next = float(target_row["y_true"])
    y_pred_rnn = float(target_row["rnn"])
    y_pred_persist = float(target_row["persistencia"])

    # Pronóstico RNN en t+1
    ax.scatter(
        [next_stamp],
        [y_pred_rnn],
        color=GREEN,
        s=140,
        marker="*",
        zorder=5,
        label=f"Pronóstico RNN ŷ(t+1) = {int(round(y_pred_rnn))} bicis",
    )

    # Valor real en t+1
    ax.scatter(
        [next_stamp],
        [y_true_next],
        color=RED,
        s=100,
        marker="o",
        edgecolor=INK,
        zorder=4,
        label=f"Demanda real y(t+1) = {int(y_true_next)} bicis",
    )

    # Persistencia en t+1
    ax.scatter(
        [next_stamp],
        [y_pred_persist],
        color=ORANGE,
        s=90,
        marker="s",
        zorder=3,
        label=f"Persistencia = {int(y_pred_persist)} bicis",
    )

    # Anotación del error en t+1
    err_rnn = abs(y_true_next - y_pred_rnn)
    ax.annotate(
        f"Error RNN: {err_rnn:.0f} bicis",
        xy=(next_stamp, y_pred_rnn),
        xytext=(15, 15),
        textcoords="offset points",
        fontsize=8.5,
        fontweight="bold",
        color=GREEN,
        arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.2),
    )

    ax.set_ylabel("Demanda de bicicletas (alquileres/hora)", fontsize=9.5)
    ax.set_title(
        f"Simulación Operativa en Tiempo Real: Pronóstico para {next_stamp:%Y-%m-%d %H:00}",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b %H:%M"))
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    csv_file = RESULTS_DIR / "test_predictions.csv"
    if csv_file.exists():
        df = pd.read_csv(csv_file)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        hist, row = get_demo_slice(df, target_idx=100, window_hours=24)
        print("Slice de prueba obtenido con éxito. Filas historial:", len(hist))
        alert = check_weather_alerts(row)
        print("Alerta detectada:", alert["level"])
        for m in alert["messages"]:
            print(" -", m)
    else:
        print("No se encontró test_predictions.csv")
