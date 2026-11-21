"""Bucle de entrenamiento reproducible y plan de hiperparámetros para la RNN.

Implementa las Etapas 10 y 11 (secciones 4.5 y 4.6 del informe):
- Configuración de aprendizaje: MSE, Adam, tamaño de batch, clipping de gradientes,
  early stopping y semilla fija (SEED = 42).
- Construcción de tensores temporales sin fuga (escalador ajustado solo en Train).
- Función `train_rnn(...)` lista para que la Parte 4 (`experiments.py`) ejecute el
  grid completo 3x3 fuera de la app y guarde los resultados en `results/`.
"""
from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_GRAD_CLIP,
    DEFAULT_LR,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_NUM_LAYERS,
    DEFAULT_PATIENCE,
    FEATURES,
    HIDDEN_SIZES,
    SEED,
    TARGET,
    TRAIN_FRAC,
    VAL_FRAC,
    WINDOW_SIZES,
    set_seeds,
)
from data import load_hour
from features import add_cyclic_features
from metrics import mae, mase, rmse, rnn_param_count
from model import SimpleRNNForecaster, count_parameters
from preprocessing import TemporalMinMaxScaler, create_sliding_windows, split_dataframe


@dataclass
class TrainConfig:
    """Hiperparámetros de entrenamiento compartidos por toda la rejilla (Tabla 8)."""

    input_size: int = len(FEATURES)
    num_layers: int = DEFAULT_NUM_LAYERS
    dropout: float = 0.0
    lr: float = DEFAULT_LR
    batch_size: int = DEFAULT_BATCH_SIZE
    max_epochs: int = DEFAULT_MAX_EPOCHS
    patience: int = DEFAULT_PATIENCE
    min_delta: float = 1e-4
    grad_clip: Optional[float] = DEFAULT_GRAD_CLIP
    seed: int = SEED
    scale_target: bool = True
    use_past_context: bool = True
    clip_non_negative: bool = True
    device: Optional[str] = None

    def resolved_device(self) -> str:
        if self.device is not None:
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["device"] = self.resolved_device()
        return d


def training_config_table(config: Optional[TrainConfig] = None) -> pd.DataFrame:
    """Tabla 8 del informe: configuración de entrenamiento y su justificación técnica."""
    cfg = config or TrainConfig()
    rows = [
        (
            "Función de pérdida",
            "MSE (Error Cuadrático Medio)",
            "Penaliza más los errores grandes en horas punta y ofrece gradientes suaves para regresión continua.",
        ),
        (
            "Optimizador",
            "Adam",
            "Adapta la tasa de aprendizaje por parámetro; converge con rapidez y estabilidad en redes recurrentes.",
        ),
        (
            "Tasa de aprendizaje (lr)",
            f"{cfg.lr:g}",
            "Valor estándar en Adam que equilibra velocidad de convergencia y estabilidad sin oscilaciones.",
        ),
        (
            "Tamaño del batch (B)",
            str(cfg.batch_size),
            "Compromiso entre estimación estable del gradiente y eficiencia en CPU/GPU sobre ~12 100 ventanas.",
        ),
        (
            "Épocas máximas / early stopping",
            f"{cfg.max_epochs} épocas (paciencia = {cfg.patience}, Δ = {cfg.min_delta:g})",
            "Detiene el entrenamiento cuando la pérdida en validación deja de mejorar, previniendo sobreajuste.",
        ),
        (
            "Clipping de gradientes",
            f"Norma L2 máx. = {cfg.grad_clip}" if cfg.grad_clip is not None else "Desactivado",
            "Evita explosión de gradientes durante la retropropagación en el tiempo (BPTT), clave para L = 48.",
        ),
        (
            "Escalado del objetivo",
            "MinMax ajustado solo en Train (inversión para MAE/RMSE/MASE)"
            if cfg.scale_target
            else "Sin escalar (escala original)",
            "Mantiene entradas y objetivo en [0, 1] durante MSE/BPTT y evalúa siempre en bicicletas/hora reales.",
        ),
        (
            "Semilla aleatoria",
            f"SEED = {cfg.seed}",
            "Garantiza reproducibilidad de inicialización de pesos, orden de mini-lotes y resultados.",
        ),
    ]
    return pd.DataFrame(rows, columns=["Elemento", "Valor", "Justificación"])


def build_hyperparameter_grid(
    window_sizes: List[int] = WINDOW_SIZES,
    hidden_sizes: List[int] = HIDDEN_SIZES,
    input_size: int = len(FEATURES),
    num_layers: int = DEFAULT_NUM_LAYERS,
    n_train: int = 12165,
) -> pd.DataFrame:
    """Construye la rejilla 3×3 (C1..C9) del plan de hiperparámetros (Tabla 9)."""
    rows = []
    idx = 1
    for L in window_sizes:
        for H in hidden_sizes:
            cid = f"C{idx}"
            if L == 12 and H == 16:
                exp_ref = "Exp. C · Memoria corta"
            elif L in (24, 48) and H in (32, 64):
                exp_ref = "Exp. D · Dependencias más largas"
            else:
                exp_ref = "Rejilla intermedia (control)"

            n_win_train = n_train - L
            rows.append(
                {
                    "ID": cid,
                    "Ventana L (h)": L,
                    "Hidden H": H,
                    "Capas": num_layers,
                    "Parámetros": rnn_param_count(input_size, H, num_layers),
                    "Tensor X_train": f"({n_win_train:,}, {L}, {input_size})",
                    "Memoria X_train (MB)": round(n_win_train * L * input_size * 4 / (1024 ** 2), 2),
                    "Rol experimental": exp_ref,
                }
            )
            idx += 1
    return pd.DataFrame(rows)


def build_window_datasets(
    df: pd.DataFrame,
    L: int,
    feature_cols: List[str] = FEATURES,
    target_col: str = TARGET,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    use_past_context: bool = True,
    scale_target: bool = True,
) -> Dict[str, Any]:
    """Genera tensores (N, L, F) de entrenamiento, validación y prueba sin fuga temporal.

    - El escalador MinMax se ajusta EXCLUSIVAMENTE con `train_df`.
    - Si `use_past_context=True`, las primeras L observaciones de `val` toman contexto
      estrictamente pasado del final de `train` (y `test` del final de `val`), nunca futuro.
      Así, todas las longitudes L ∈ {12, 24, 48} y todos los baselines se evalúan exactamente
      sobre las mismas 2 607 horas de validación y 2 607 horas de prueba.
    """
    train_df, val_df, test_df = split_dataframe(df, train_frac=train_frac, val_frac=val_frac)

    scaler = TemporalMinMaxScaler(cols=feature_cols)
    train_scaled = scaler.fit_transform(train_df)
    val_scaled = scaler.transform(val_df)
    test_scaled = scaler.transform(test_df)

    # Entrenamiento: siempre estrictamente dentro de train_df
    X_train, y_train_raw = create_sliding_windows(
        train_scaled, train_df, L=L, feature_cols=feature_cols, target_col=target_col
    )

    # Validación y Prueba
    if use_past_context:
        val_ctx_scaled = pd.concat([train_scaled.iloc[-L:], val_scaled], ignore_index=True)
        val_ctx_raw = pd.concat([train_df.iloc[-L:], val_df], ignore_index=True)
        X_val, y_val_raw = create_sliding_windows(
            val_ctx_scaled, val_ctx_raw, L=L, feature_cols=feature_cols, target_col=target_col
        )
        val_timestamps = val_df["timestamp"].reset_index(drop=True)

        test_ctx_scaled = pd.concat([val_scaled.iloc[-L:], test_scaled], ignore_index=True)
        test_ctx_raw = pd.concat([val_df.iloc[-L:], test_df], ignore_index=True)
        X_test, y_test_raw = create_sliding_windows(
            test_ctx_scaled, test_ctx_raw, L=L, feature_cols=feature_cols, target_col=target_col
        )
        test_timestamps = test_df["timestamp"].reset_index(drop=True)
    else:
        X_val, y_val_raw = create_sliding_windows(
            val_scaled, val_df, L=L, feature_cols=feature_cols, target_col=target_col
        )
        val_timestamps = val_df["timestamp"].iloc[L:].reset_index(drop=True)

        X_test, y_test_raw = create_sliding_windows(
            test_scaled, test_df, L=L, feature_cols=feature_cols, target_col=target_col
        )
        test_timestamps = test_df["timestamp"].iloc[L:].reset_index(drop=True)

    if scale_target:
        c_min, c_max = scaler.stats_[target_col]
        denom = c_max - c_min if c_max > c_min else 1.0
        y_train_scaled = ((y_train_raw - c_min) / denom).astype(np.float32)
        y_val_scaled = ((y_val_raw - c_min) / denom).astype(np.float32)
        y_test_scaled = ((y_test_raw - c_min) / denom).astype(np.float32)
    else:
        y_train_scaled = y_train_raw.astype(np.float32)
        y_val_scaled = y_val_raw.astype(np.float32)
        y_test_scaled = y_test_raw.astype(np.float32)

    return {
        "X_train": X_train,
        "y_train_scaled": y_train_scaled,
        "y_train_raw": y_train_raw,
        "X_val": X_val,
        "y_val_scaled": y_val_scaled,
        "y_val_raw": y_val_raw,
        "X_test": X_test,
        "y_test_scaled": y_test_scaled,
        "y_test_raw": y_test_raw,
        "scaler": scaler,
        "y_train_series_raw": train_df[target_col].to_numpy(dtype=float),
        "val_timestamps": val_timestamps,
        "test_timestamps": test_timestamps,
    }


@torch.no_grad()
def predict(
    model: nn.Module,
    X: np.ndarray,
    scaler: Optional[TemporalMinMaxScaler] = None,
    batch_size: int = 256,
    device: str = "cpu",
    scale_target: bool = True,
    clip_non_negative: bool = True,
) -> np.ndarray:
    """Realiza inferencia por lotes y revierte el escalado del objetivo a bicicletas/hora."""
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=batch_size, shuffle=False)
    chunks = []
    for (xb,) in loader:
        xb = xb.to(device)
        preds = model(xb).squeeze(-1).cpu().numpy()
        chunks.append(preds)
    y_hat = np.concatenate(chunks, axis=0)
    if scale_target and scaler is not None:
        y_hat = scaler.inverse_transform_target(y_hat, target_col=TARGET)
    if clip_non_negative:
        y_hat = np.clip(y_hat, 0.0, None)
    return y_hat


def evaluate_model(
    model: nn.Module,
    X: np.ndarray,
    y_raw: np.ndarray,
    y_train_series_raw: np.ndarray,
    scaler: Optional[TemporalMinMaxScaler] = None,
    batch_size: int = 256,
    device: str = "cpu",
    scale_target: bool = True,
    clip_non_negative: bool = True,
) -> Dict[str, Any]:
    """Calcula MAE, RMSE y MASE en escala original (bicicletas/hora)."""
    y_pred = predict(
        model=model,
        X=X,
        scaler=scaler,
        batch_size=batch_size,
        device=device,
        scale_target=scale_target,
        clip_non_negative=clip_non_negative,
    )
    return {
        "mae": mae(y_raw, y_pred),
        "rmse": rmse(y_raw, y_pred),
        "mase": mase(y_raw, y_pred, y_train_series_raw),
        "y_pred": y_pred,
    }


def train_rnn(
    L: int,
    hidden_size: int,
    df: Optional[pd.DataFrame] = None,
    config: Optional[TrainConfig] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Entrena una `SimpleRNNForecaster` para (L, H) usando solo Train y validando en Val.

    Nunca accede al conjunto de prueba (reservado exclusivamente para la evaluación final
    en la Parte 4). Devuelve el mejor modelo según pérdida MSE en validación, historial de
    curvas de pérdida, métricas de validación en escala original y tiempo de entrenamiento.
    """
    cfg = config or TrainConfig()
    set_seeds(cfg.seed)
    device = cfg.resolved_device()

    if df is None:
        df = add_cyclic_features(load_hour())

    data = build_window_datasets(
        df=df,
        L=L,
        feature_cols=FEATURES,
        target_col=TARGET,
        use_past_context=cfg.use_past_context,
        scale_target=cfg.scale_target,
    )

    train_ds = TensorDataset(
        torch.from_numpy(data["X_train"]),
        torch.from_numpy(data["y_train_scaled"]).unsqueeze(-1),
    )
    val_ds = TensorDataset(
        torch.from_numpy(data["X_val"]),
        torch.from_numpy(data["y_val_scaled"]).unsqueeze(-1),
    )

    # Nota: solo se barajan mini-lotes de ventanas ya formadas dentro del tramo de
    # entrenamiento (modelo stateless, h_0 = 0); nunca se mezclan Train, Val ni Test.
    gen = torch.Generator()
    gen.manual_seed(cfg.seed)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, generator=gen)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = SimpleRNNForecaster(
        input_size=cfg.input_size,
        hidden_size=hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)
    n_params = count_parameters(model)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0

    t0 = time.perf_counter()
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        n_train_seen = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            if cfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(xb)
            n_train_seen += len(xb)

        train_mse = train_loss_sum / max(1, n_train_seen)

        model.eval()
        val_loss_sum = 0.0
        n_val_seen = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss_sum += float(loss.item()) * len(xb)
                n_val_seen += len(xb)
        val_mse = val_loss_sum / max(1, n_val_seen)

        history["train_loss"].append(train_mse)
        history["val_loss"].append(val_mse)

        if val_mse < best_val_loss - cfg.min_delta:
            best_val_loss = val_mse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if verbose and (epoch == 1 or epoch % 5 == 0 or no_improve == cfg.patience):
            print(
                f"[L={L:2d}, H={hidden_size:2d}] Época {epoch:02d}/{cfg.max_epochs} | "
                f"train_mse={train_mse:.5f} | val_mse={val_mse:.5f}"
            )

        if no_improve >= cfg.patience:
            break

    train_time_sec = time.perf_counter() - t0
    model.load_state_dict(best_state)

    val_eval = evaluate_model(
        model=model,
        X=data["X_val"],
        y_raw=data["y_val_raw"],
        y_train_series_raw=data["y_train_series_raw"],
        scaler=data["scaler"],
        batch_size=256,
        device=device,
        scale_target=cfg.scale_target,
        clip_non_negative=cfg.clip_non_negative,
    )

    return {
        "L": L,
        "hidden_size": hidden_size,
        "num_layers": cfg.num_layers,
        "n_params": n_params,
        "model": model,
        "best_state_dict": best_state,
        "scaler": data["scaler"],
        "history": history,
        "best_epoch": best_epoch,
        "epochs_run": len(history["train_loss"]),
        "best_val_mse": best_val_loss,
        "train_time_sec": train_time_sec,
        "val_metrics": {
            "mae": val_eval["mae"],
            "rmse": val_eval["rmse"],
            "mase": val_eval["mase"],
        },
        "val_preds": val_eval["y_pred"],
        "val_true": data["y_val_raw"],
        "val_timestamps": data["val_timestamps"],
        "config": cfg.to_dict(),
    }


if __name__ == "__main__":
    # Verificación rápida (smoke test) de la configuración C1 (memoria corta: L=12, H=16)
    res = train_rnn(L=12, hidden_size=16, verbose=True)
    print(
        f"\nC1 (L=12, H=16) | Parámetros: {res['n_params']} | "
        f"Mejor época: {res['best_epoch']}/{res['epochs_run']} | "
        f"Tiempo: {res['train_time_sec']:.2f}s | "
        f"Val MAE: {res['val_metrics']['mae']:.2f} | "
        f"Val RMSE: {res['val_metrics']['rmse']:.2f} | "
        f"Val MASE: {res['val_metrics']['mase']:.3f}"
    )
