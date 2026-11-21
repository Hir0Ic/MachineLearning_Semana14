"""Configuración central: rutas, semilla y constantes del experimento."""
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "hour.csv"
DIAGRAM_PATH = ROOT / "assets" / "arquitectura_rnn.png"
RESULTS_DIR = ROOT / "results"

SEED = 42

# Particiones cronológicas (sin mezclar): entrenamiento / validación / prueba
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

# Características F de cada paso de la ventana (sección 3.3)
FEATURES = [
    "cnt", "temp", "hum", "windspeed",   # historia y clima observados hasta t
    "hr_sin", "hr_cos",                  # hora del día (calendario)
    "workingday", "holiday",             # calendario
]
TARGET = "cnt"  # y = cnt(t+1)

# Plan de hiperparámetros y configuración base de entrenamiento (secciones 4.3 a 4.6)
WINDOW_SIZES = [12, 24, 48]
HIDDEN_SIZES = [16, 32, 64]
DEFAULT_NUM_LAYERS = 1
DEFAULT_LR = 1e-3
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_EPOCHS = 50
DEFAULT_PATIENCE = 8
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_SEASONAL_DAYS = 7


def set_seeds(seed: int = SEED) -> None:
    """Fija las semillas de Python, NumPy y PyTorch (si está instalado) para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
