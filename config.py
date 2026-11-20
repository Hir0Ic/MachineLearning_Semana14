"""Configuración central: rutas, semilla y constantes del experimento."""
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "hour.csv"
DIAGRAM_PATH = ROOT / "assets" / "arquitectura_rnn.png"

SEED = 42

# Particiones cronológicas (sin mezclar): entrenamiento / validación / prueba
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

# Características F de cada paso de la ventana (ajústelas a su sección 3.3)
FEATURES = [
    "cnt", "temp", "hum", "windspeed",   # historia y clima observados hasta t
    "hr_sin", "hr_cos",                  # hora del día (calendario)
    "workingday", "holiday",             # calendario
]
TARGET = "cnt"  # y = cnt(t+1)


def set_seeds(seed: int = SEED) -> None:
    """Fija las semillas de Python y NumPy (agregue torch.manual_seed si entrena en PyTorch)."""
    random.seed(seed)
    np.random.seed(seed)
