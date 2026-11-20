"""Carga, ingeniería de características, ventanas y partición temporal."""
import numpy as np
import pandas as pd

from config import DATA_PATH, FEATURES, TRAIN_FRAC, VAL_FRAC

# Descripción de variables para la ficha del dataset (Tabla 3 del informe)
VARIABLES = [
    ("dteday, hr", "Fecha y hora del registro", "dteday: AAAA-MM-DD; hr: 0 a 23", "Orden temporal; índice horario", "Sí"),
    ("cnt", "Total de alquileres", "Entero ≥ 0 (bicicletas/hora)", "Objetivo y rezagos permitidos", "Solo hasta t"),
    ("casual, registered", "Alquileres por tipo de usuario", "Enteros ≥ 0; casual + registered = cnt", "EXCLUIDAS de los predictores", "No (suman cnt)"),
    ("temp, hum, windspeed", "Variables meteorológicas", "Normalizadas en [0, 1]", "Predictores", "Solo valores hasta t"),
    ("holiday, weekday, workingday", "Calendario", "holiday y workingday: 0/1; weekday: 0 a 6", "Predictores", "Sí (calendario)"),
    ("season, mnth", "Estación y mes", "season: 1 a 4; mnth: 1 a 12", "Candidatas; derivables de dteday", "Sí (calendario)"),
    ("weathersit", "Estado del tiempo", "Categórica de 1 a 4", "Predictor candidato", "Solo valores hasta t"),
    ("atemp", "Sensación térmica", "Normalizada en [0, 1]", "Descartada: redundante con temp", "Solo valores hasta t"),
    ("yr", "Año del registro", "0 = 2011; 1 = 2012", "Descartada: derivable de dteday", "Sí (calendario)"),
    ("instant", "Índice secuencial", "Entero consecutivo", "Descartada: identificador", "Sí (no aporta)"),
]


def load_hour(path=DATA_PATH) -> pd.DataFrame:
    """Lee hour.csv, ordena cronológicamente y construye un timestamp."""
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["dteday"]) + pd.to_timedelta(df["hr"], unit="h")
    return df.sort_values("timestamp").reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega la codificación cíclica de la hora (sin/cos)."""
    out = df.copy()
    out["hr_sin"] = np.sin(2 * np.pi * out["hr"] / 24)
    out["hr_cos"] = np.cos(2 * np.pi * out["hr"] / 24)
    return out


def temporal_split(n: int, train_frac: float = TRAIN_FRAC, val_frac: float = VAL_FRAC):
    """Índices de corte cronológicos: [0, i_tr) entrenamiento, [i_tr, i_va) validación, resto prueba."""
    i_tr = int(n * train_frac)
    i_va = int(n * (train_frac + val_frac))
    return i_tr, i_va


def fit_minmax(df_train: pd.DataFrame, cols=FEATURES) -> dict:
    """Mínimos y máximos calculados SOLO con entrenamiento (evita fuga por escalado)."""
    return {c: (float(df_train[c].min()), float(df_train[c].max())) for c in cols}


def apply_minmax(df: pd.DataFrame, scaler: dict) -> pd.DataFrame:
    out = df.copy()
    for c, (lo, hi) in scaler.items():
        out[c] = (out[c] - lo) / (hi - lo) if hi > lo else 0.0
    return out


def make_windows(df: pd.DataFrame, L: int, features=FEATURES, target="cnt"):
    """Ventanas deslizantes.

    X[i] = filas [i, i+L)  -> tensor (N, L, F), valores hasta t = i+L-1
    y[i] = cnt en la fila i+L  -> cnt(t+1)
    """
    values = df[features].to_numpy(dtype=np.float32)
    tgt = df[target].to_numpy(dtype=np.float32)
    n = len(df) - L
    X = np.stack([values[i : i + L] for i in range(n)])
    y = tgt[L : L + n]
    return X, y
