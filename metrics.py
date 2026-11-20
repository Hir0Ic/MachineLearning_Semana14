"""Métricas de evaluación, líneas base y conteo de parámetros."""
import numpy as np


def mae(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.mean(np.abs(y - yhat)))


def rmse(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def mae_naive_train(y_train) -> float:
    """MAE de la persistencia a un paso sobre entrenamiento: mean |y_t - y_(t-1)|."""
    y_train = np.asarray(y_train, float)
    return float(np.mean(np.abs(np.diff(y_train))))


def mase(y, yhat, y_train) -> float:
    """MAE del modelo / MAE ingenuo (persistencia) calculado solo con entrenamiento."""
    return mae(y, yhat) / mae_naive_train(y_train)


def persistence_forecast(series: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Línea base A: ŷ(t+1) = y(t). `idx` son los índices t."""
    return np.asarray(series, float)[idx]


def same_hour_mean_forecast(series: np.ndarray, idx: np.ndarray, days: int = 7) -> np.ndarray:
    """Línea base B: promedio de la misma hora de los últimos `days` días.

    Para predecir la hora t+1 usa y(t+1-24k), k = 1..days (todos <= t).
    """
    s = np.asarray(series, float)
    preds = []
    for t in idx:
        lags = [t + 1 - 24 * k for k in range(1, days + 1) if t + 1 - 24 * k >= 0]
        preds.append(np.mean(s[lags]) if lags else s[t])
    return np.array(preds)


def rnn_param_count(F: int, H: int, layers: int = 1) -> int:
    """Parámetros de nn.RNN (dos sesgos por capa) + cabeza lineal (H + 1).

    Capa 1: H*F + H*H + 2H. Capas siguientes reciben H entradas: H*H + H*H + 2H.
    """
    total = H * F + H * H + 2 * H
    total += (layers - 1) * (H * H + H * H + 2 * H)
    return total + H + 1
