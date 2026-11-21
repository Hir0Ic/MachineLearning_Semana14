"""Modelos de referencia (baselines) estrictamente causales para pronóstico t+1.

Implementa la Etapa 08 (sección 4.3 del informe):
- A. Persistencia: y_hat(t+1) = cnt(t)
- B. Promedio móvil estacional: promedio de la misma hora en los últimos k días recientes
- C. Regresión Ridge sobre rezagos (opcional): combinación lineal causal con rezagos hasta t

Todos los modelos usan exclusivamente observaciones pasadas (<= t) y permiten
evaluar sobre las mismas marcas temporales de validación o prueba que la RNN.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from config import DEFAULT_SEASONAL_DAYS, TARGET, TRAIN_FRAC, VAL_FRAC
from metrics import (
    mae,
    mase,
    persistence_forecast,
    rmse,
    same_hour_mean_forecast,
)
from preprocessing import temporal_split_indices

# Rezagos causales (en horas respecto al instante objetivo t+1) para la regresión C:
# lag=1 -> cnt(t), lag=2 -> cnt(t-1), lag=3 -> cnt(t-2),
# lag=24 -> cnt(t-23) [misma hora día anterior], lag=168 -> cnt(t-167) [misma hora semana anterior]
DEFAULT_LAGS = (1, 2, 3, 24, 168)
DEFAULT_EXOG_AT_T = ("temp", "hum", "windspeed", "hr_sin", "hr_cos", "workingday", "holiday")


def get_split_target_indices(
    n_samples: int,
    split: str = "val",
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    offset: int = 0,
) -> np.ndarray:
    """Devuelve los índices de las observaciones objetivo y(t+1) para el tramo indicado.

    Para cada índice `target_idx`, el último instante observado disponible es `t = target_idx - 1`.
    Con `offset = 0`, validación cubre las 2 607 filas de [i_tr, i_va) y prueba las 2 607 de [i_va, n),
    exactamente alineado con ventanas que usan contexto histórico pasado del tramo anterior.
    """
    i_tr, i_va = temporal_split_indices(n_samples, train_frac, val_frac)
    if split == "train":
        start = max(1, offset)
        return np.arange(start, i_tr)
    if split == "val":
        return np.arange(i_tr + offset, i_va)
    if split == "test":
        return np.arange(i_va + offset, n_samples)
    raise ValueError(f"Partición desconocida: {split!r}. Use 'train', 'val' o 'test'.")


def persistence_predictions(series: np.ndarray, target_indices: np.ndarray) -> np.ndarray:
    """Baseline A · Persistencia: y_hat(t+1) = cnt(t)."""
    t_indices = np.asarray(target_indices, dtype=int) - 1
    if np.any(t_indices < 0):
        raise ValueError("target_indices debe ser >= 1 para contar con cnt(t).")
    return persistence_forecast(series, t_indices)


def seasonal_mean_predictions(
    series: np.ndarray,
    target_indices: np.ndarray,
    days: int = DEFAULT_SEASONAL_DAYS,
) -> np.ndarray:
    """Baseline B · Promedio estacional: media de la misma hora en los últimos `days` días.

    Para predecir la fila `idx` (= t+1), promedia `series[idx - 24*k]` para k = 1..days,
    todos estrictamente <= t. Con `days = 1` equivale a la misma hora del día anterior.
    """
    if days < 1:
        raise ValueError("`days` debe ser un entero >= 1.")
    t_indices = np.asarray(target_indices, dtype=int) - 1
    if np.any(t_indices < 0):
        raise ValueError("target_indices debe ser >= 1.")
    return same_hour_mean_forecast(series, t_indices, days=days)


class LagRegressionBaseline:
    """Baseline C (opcional) · Regresión lineal Ridge sobre rezagos y variables hasta t.

    Se ajusta exclusivamente con el conjunto de entrenamiento (sin fuga de información)
    y utiliza únicamente rezagos pasados de `cnt` y covariables observadas en `t`.
    """

    def __init__(
        self,
        lags: Tuple[int, ...] = DEFAULT_LAGS,
        exog_cols: Tuple[str, ...] = DEFAULT_EXOG_AT_T,
        alpha: float = 1.0,
        clip_non_negative: bool = True,
    ):
        self.lags = tuple(lags)
        self.exog_cols = tuple(exog_cols)
        self.alpha = float(alpha)
        self.clip_non_negative = clip_non_negative
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def _build_design_matrix(self, df: pd.DataFrame, target_indices: np.ndarray) -> np.ndarray:
        """Construye la matriz de predictores usando exclusivamente índices <= target_idx - 1."""
        s = df[TARGET].to_numpy(dtype=float)
        idx = np.asarray(target_indices, dtype=int)
        t_idx = idx - 1

        cols = [s[idx - lag] for lag in self.lags]
        for col in self.exog_cols:
            if col in df.columns:
                cols.append(df[col].to_numpy(dtype=float)[t_idx])
        return np.column_stack(cols)

    def fit(self, df: pd.DataFrame, train_end: int) -> "LagRegressionBaseline":
        """Ajusta los coeficientes Ridge únicamente con observaciones objetivo en [max_lag, train_end)."""
        max_lag = max(self.lags)
        train_targets = np.arange(max_lag, train_end)
        X_raw = self._build_design_matrix(df, train_targets)
        y_train = df[TARGET].to_numpy(dtype=float)[train_targets]

        # Estandarización calculada únicamente en entrenamiento
        self.mean_ = X_raw.mean(axis=0)
        self.std_ = np.where(X_raw.std(axis=0) > 1e-8, X_raw.std(axis=0), 1.0)
        X_std = (X_raw - self.mean_) / self.std_

        # Ridge cerrado sin penalizar el intercepto
        X_aug = np.column_stack([np.ones(len(X_std)), X_std])
        reg = np.eye(X_aug.shape[1]) * self.alpha
        reg[0, 0] = 0.0
        theta = np.linalg.solve(X_aug.T @ X_aug + reg, X_aug.T @ y_train)
        self.intercept_ = float(theta[0])
        self.coef_ = theta[1:]
        return self

    def predict(self, df: pd.DataFrame, target_indices: np.ndarray) -> np.ndarray:
        """Genera pronósticos para `target_indices` sin usar información futura."""
        if self.coef_ is None or self.mean_ is None or self.std_ is None:
            raise RuntimeError("Debe ejecutar `fit` antes de `predict`.")
        X_raw = self._build_design_matrix(df, target_indices)
        X_std = (X_raw - self.mean_) / self.std_
        preds = self.intercept_ + X_std @ self.coef_
        if self.clip_non_negative:
            preds = np.clip(preds, 0.0, None)
        return preds


def evaluate_baselines(
    df: pd.DataFrame,
    split: str = "val",
    seasonal_days: int = DEFAULT_SEASONAL_DAYS,
    include_lag_reg: bool = True,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    offset: int = 0,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], np.ndarray, pd.Series]:
    """Evalúa los modelos de referencia en `split` ('val' o 'test').

    Retorna:
        table_df: DataFrame con MAE, RMSE, MASE y mejora porcentual respecto a persistencia.
        preds: Diccionario {nombre_modelo: vector de predicciones}.
        y_true: Valores reales observados cnt(t+1).
        timestamps: Serie con las marcas temporales correspondientes a cada y_true.
    """
    series = df[TARGET].to_numpy(dtype=float)
    i_tr, _ = temporal_split_indices(len(df), train_frac, val_frac)
    y_train_ref = series[:i_tr]

    target_idx = get_split_target_indices(
        len(df), split=split, train_frac=train_frac, val_frac=val_frac, offset=offset
    )
    y_true = series[target_idx]
    timestamps = df["timestamp"].iloc[target_idx].reset_index(drop=True)

    preds: Dict[str, np.ndarray] = {
        "A. Persistencia: cnt(t)": persistence_predictions(series, target_idx),
        f"B. Promedio estacional (k={seasonal_days} días)": seasonal_mean_predictions(
            series, target_idx, days=seasonal_days
        ),
    }

    if seasonal_days != 1:
        preds["B0. Misma hora día anterior (k=1 día)"] = seasonal_mean_predictions(
            series, target_idx, days=1
        )

    if include_lag_reg:
        lag_model = LagRegressionBaseline().fit(df, train_end=i_tr)
        preds["C. Regresión Ridge sobre rezagos (opcional)"] = lag_model.predict(df, target_idx)

    persist_mae = mae(y_true, preds["A. Persistencia: cnt(t)"])

    rows = []
    for name, yhat in preds.items():
        m = mae(y_true, yhat)
        r = rmse(y_true, yhat)
        ms = mase(y_true, yhat, y_train_ref)
        imp = 0.0 if name.startswith("A.") else (1.0 - m / persist_mae) * 100.0
        rows.append(
            {
                "Modelo": name,
                "MAE": m,
                "RMSE": r,
                "MASE": ms,
                "Mejora MAE vs. Persistencia (%)": imp,
            }
        )

    table_df = pd.DataFrame(rows)
    return table_df, preds, y_true, timestamps
