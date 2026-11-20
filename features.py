"""Ingeniería y definición de características para el pronóstico de demanda."""
import numpy as np
import pandas as pd

from config import FEATURES, TARGET

# Variables candidatas y excluidas para documentar la selección
EXCLUDED_LEAKAGE = ["casual", "registered"]
EXCLUDED_REDUNDANT = ["atemp", "instant", "yr"]

def add_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Codificación cíclica con seno y coseno para hora del día y mes.
    
    Permite a la red capturar la continuidad periódica (por ejemplo, hora 23 y hora 0 contiguas).
    """
    out = df.copy()
    # Hora del día (periodo 24 horas)
    out["hr_sin"] = np.sin(2 * np.pi * out["hr"] / 24.0)
    out["hr_cos"] = np.cos(2 * np.pi * out["hr"] / 24.0)
    
    # Mes del año (periodo 12 meses)
    out["mnth_sin"] = np.sin(2 * np.pi * (out["mnth"] - 1) / 12.0)
    out["mnth_cos"] = np.cos(2 * np.pi * (out["mnth"] - 1) / 12.0)
    return out


def prepare_features_and_target(df: pd.DataFrame, feature_cols=FEATURES, target_col=TARGET):
    """Extrae las variables predictoras X y el objetivo y = cnt(t+1).
    
    Asegura que no exista fuga de información de t+1 en las variables X.
    """
    df_features = add_cyclic_features(df)
    
    # Verificación de que no existan variables prohibidas
    for leak in EXCLUDED_LEAKAGE:
        if leak in feature_cols:
            raise ValueError(f"Fuga detectada: {leak} no debe estar en las características predictoras.")
            
    X_df = df_features[feature_cols].copy()
    # El objetivo es la demanda observada en la siguiente hora t+1
    y_series = df[target_col].shift(-1)
    
    return X_df, y_series
