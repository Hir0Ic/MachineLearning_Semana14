# Pronóstico horario de alquiler de bicicletas con una RNN simple · Grupo 03

App interactiva de Streamlit y módulos reproducibles en Python/PyTorch para pronóstico horario ($t+1$) de demanda de bicicletas compartidas usando una Red Neuronal Recurrente (RNN) simple.

## Ejecución

```bash
# 1. Crear el entorno virtual
python -m venv .venv

# 2. Activar el entorno virtual
# En Windows (PowerShell):
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Iniciar la aplicación
streamlit run app.py
```

## Estructura del Proyecto

| Archivo | Parte | Contenido |
|---|---|---|
| `app.py` | 1–3 | Interfaz interactiva de Streamlit con una pestaña por integrante (Partes 1, 2 y 3) |
| `config.py` | 1 / 3 | Semilla fija (`SEED = 42`), rutas, particiones 70/15/15 y constantes de entrenamiento/rejilla |
| `data.py` | 1 | Funciones de carga de `hour.csv` y compatibilidad de datos |
| `metrics.py` | 1 | Métricas de evaluación (MAE, RMSE, MASE), fórmulas base y conteo analítico de parámetros RNN |
| `features.py` | 2 | Definición de $y = \text{cnt}(t+1)$, características cíclicas y exclusión de variables con fuga |
| `preprocessing.py` | 2 | Partición cronológica 70/15/15, escalador MinMax ajustado solo en Train y ventanas $(N, L, F)$ |
| `baselines.py` | 3 | Modelos de referencia causales en validación/prueba: persistencia, promedio estacional ($k$ días) y regresión Ridge sobre rezagos |
| `model.py` | 3 | Arquitectura PyTorch `SimpleRNNForecaster` (1–2 capas `nn.RNN(tanh)` + cabeza lineal), tabla de capas y conteo exacto de parámetros |
| `train.py` | 3 | Configuración `TrainConfig` (MSE, Adam, batch, clipping, early stopping, semilla), bucle reproducible `train_rnn(...)` y rejilla 3×3 (`C1`–`C9`) |
| `data/hour.csv` | 1 | Dataset UCI Bike Sharing (ID 275) |
| `assets/arquitectura_rnn.png` | 1 | Diagrama de la arquitectura RNN muchos a uno |

## Secciones Cubiertas

- **Parte 1 (del 1 al 3.2):** Introducción, marco teórico, fuga empírica de información (`casual + registered = cnt`), arquitectura RNN, métricas y ficha del dataset.
- **Parte 2 (del 3.3 al 4.2):**
  - $y = \text{cnt}(t+1)$ y selección de características finales ($F=8$).
  - EDA: patrones horarios, picos de hora punta, días laborales vs. fines de semana y estacionalidad.
  - Serie temporal completa coloreada con partición cronológica (70 % / 15 % / 15 %).
  - Escalado MinMax ajustado exclusivamente con entrenamiento y ventanas deslizantes $L \in \{12, 24, 48\}$.
- **Parte 3 (del 4.3 al 4.6):**
  - Modelos de referencia causales (persistencia, promedio estacional y regresión Ridge sobre rezagos) evaluados en validación.
  - Arquitectura concreta `SimpleRNNForecaster` en PyTorch con selector de $H \in \{16, 32, 64\}$, $L \in \{12, 24, 48\}$, 1–2 capas y verificación de parámetros entrenables.
  - Configuración de aprendizaje reproducible (MSE, Adam, `lr=0.001`, batch `64`, clipping `1.0`, early stopping y `SEED = 42`).
  - Plan de hiperparámetros con la rejilla 3×3 (`C1`–`C9`) listo para ser entrenado fuera de la app por la Parte 4 (`experiments.py`).
