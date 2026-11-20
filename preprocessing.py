"""División temporal, escalado sin fuga y generación de tensores de ventanas (L=12, 24, 48)."""
import numpy as np
import pandas as pd
from typing import Dict, Tuple

from config import FEATURES, TARGET, TRAIN_FRAC, VAL_FRAC


def temporal_split_indices(n_samples: int, train_frac: float = TRAIN_FRAC, val_frac: float = VAL_FRAC) -> Tuple[int, int]:
    """Calcula los índices de corte para partición cronológica (sin barajar).
    
    - Entrenamiento: [0, i_tr) -> 70%
    - Validación:    [i_tr, i_va) -> 15%
    - Prueba:        [i_va, n_samples) -> 15%
    """
    i_tr = int(n_samples * train_frac)
    i_va = int(n_samples * (train_frac + val_frac))
    return i_tr, i_va


def split_dataframe(df: pd.DataFrame, train_frac: float = TRAIN_FRAC, val_frac: float = VAL_FRAC) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide un DataFrame en train, val y test respetando el orden temporal estricto."""
    i_tr, i_va = temporal_split_indices(len(df), train_frac, val_frac)
    train_df = df.iloc[:i_tr].copy()
    val_df = df.iloc[i_tr:i_va].copy()
    test_df = df.iloc[i_va:].copy()
    return train_df, val_df, test_df


class TemporalMinMaxScaler:
    """Escalador Min-Max ajustado EXCLUSIVAMENTE con el conjunto de entrenamiento.
    
    Evita la fuga de información (data leakage) hacia validación y prueba.
    """
    def __init__(self, cols=FEATURES):
        self.cols = cols
        self.stats_: Dict[str, Tuple[float, float]] = {}

    def fit(self, df_train: pd.DataFrame):
        for col in self.cols:
            c_min = float(df_train[col].min())
            c_max = float(df_train[col].max())
            self.stats_[col] = (c_min, c_max)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col, (c_min, c_max) in self.stats_.items():
            if col in out.columns:
                denom = c_max - c_min
                out[col] = (out[col] - c_min) / denom if denom > 0 else 0.0
        return out

    def fit_transform(self, df_train: pd.DataFrame) -> pd.DataFrame:
        self.fit(df_train)
        return self.transform(df_train)

    def inverse_transform_target(self, y_scaled: np.ndarray, target_col=TARGET) -> np.ndarray:
        """Revierte la escala únicamente para la variable objetivo."""
        if target_col not in self.stats_:
            return y_scaled
        c_min, c_max = self.stats_[target_col]
        return y_scaled * (c_max - c_min) + c_min


def create_sliding_windows(
    df_scaled: pd.DataFrame,
    df_raw: pd.DataFrame,
    L: int,
    feature_cols=FEATURES,
    target_col=TARGET,
) -> Tuple[np.ndarray, np.ndarray]:
    """Genera ventanas deslizantes para la RNN.
    
    Parámetros:
        df_scaled: DataFrame con características escaladas.
        df_raw: DataFrame original con valores reales (para el objetivo y sin escalar si se desea).
        L: Longitud de la ventana temporal (12, 24, 48).
        feature_cols: Lista de características de entrada F.
        target_col: Nombre de la variable objetivo (cnt).
        
    Retorna:
        X: Tensor 3D de forma (N, L, F) con la secuencia observada hasta t.
        y: Vector 1D de tamaño (N,) con el objetivo real cnt en t+1.
    """
    feat_values = df_scaled[feature_cols].to_numpy(dtype=np.float32)
    target_values = df_raw[target_col].to_numpy(dtype=np.float32)
    
    n_samples = len(df_scaled) - L
    if n_samples <= 0:
        raise ValueError(f"El tamaño de la serie ({len(df_scaled)}) es menor o igual a L={L}.")
        
    X = np.stack([feat_values[i : i + L] for i in range(n_samples)], axis=0)
    # y[i] corresponde a la observación en el paso i + L (es decir, t+1 respecto a la ventana)
    y = target_values[L : L + n_samples]
    
    return X, y
