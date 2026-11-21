# Pronóstico horario de alquiler de bicicletas con una RNN simple · Grupo 03

App interactiva de Streamlit y módulos reproducibles en Python/PyTorch para pronóstico horario ($t+1$) de demanda de bicicletas compartidas usando una Red Neuronal Recurrente (RNN) simple.

## Ejecución del Proyecto

Sigue estos pasos en tu terminal (PowerShell en Windows o Bash en Linux/macOS):

```bash
# 1. Crear el entorno virtual (usar Python 3.10 a 3.12 recomendado)
python -m venv .venv

# 2. Activar el entorno virtual
# En Windows (PowerShell):
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

# En Linux / macOS:
# source .venv/bin/activate

# 3. Instalar dependencias requeridas
pip install -r requirements.txt

# 4. (Parte 4) Ejecutar el entrenamiento de la rejilla C1–C9 y la evaluación única en prueba
# Nota: Si ya existen los archivos en results/, este paso es opcional.
python experiments.py
python evaluate.py

# 5. Iniciar la aplicación interactiva de Streamlit
streamlit run app.py
```

> **Nota de Arquitectura:** `experiments.py` y `evaluate.py` se ejecutan **fuera de la aplicación**. Las pestañas 4 y 5 leen directamente los artefactos generados en `results/`, garantizando una interfaz ágil, reactiva y libre de reentrenamientos en vivo.

---

## Estructura del Proyecto

| Archivo | Parte | Contenido |
|---|---|---|
| `app.py` | 1–5 | Interfaz interactiva de Streamlit con las 5 pestañas por integrante (Partes 1 a 5) |
| `config.py` | 1 / 3 | Semilla fija (`SEED = 42`), rutas, particiones 70/15/15 y constantes de entrenamiento/rejilla |
| `data.py` | 1 | Funciones de carga de `hour.csv` y compatibilidad de datos |
| `metrics.py` | 1 | Métricas de evaluación (MAE, RMSE, MASE), fórmulas base y conteo analítico de parámetros RNN |
| `features.py` | 2 | Definición de $y = \text{cnt}(t+1)$, características cíclicas y exclusión de variables con fuga |
| `preprocessing.py` | 2 | Partición cronológica 70/15/15, escalador MinMax ajustado solo en Train y ventanas $(N, L, F)$ |
| `baselines.py` | 3 | Modelos de referencia causales en validación/prueba: persistencia, promedio estacional ($k$ días) y regresión Ridge sobre rezagos |
| `model.py` | 3 | Arquitectura PyTorch `SimpleRNNForecaster` (1–2 capas `nn.RNN(tanh)` + cabeza lineal), tabla de capas y conteo exacto de parámetros |
| `train.py` | 3 | Configuración `TrainConfig` (MSE, Adam, batch, clipping, early stopping, semilla), bucle reproducible `train_rnn(...)` y rejilla 3×3 (`C1`–`C9`) |
| `experiments.py` | 4 | Entrena la rejilla 3×3 (C1–C9) con `train_rnn`, selecciona por menor MAE de validación y guarda modelos, historiales, predicciones de validación, entorno y configuración en `results/` |
| `evaluate.py` | 4 | Evaluación única en prueba de la configuración elegida frente a los baselines, guarda `best_model.pt`, `scaler.json`, `test_predictions.csv`, `test_metrics.csv` y figuras; funciones de gráficos reutilizadas por la app |
| `error_analysis.py` | 5 | Análisis desagregado de errores (por hora 0–23, días laborales vs. fines de semana, días pico y sensibilidad al clima) |
| `demo.py` | 5 | Simulador interactivo en tiempo real con selector de ventana (24/48 h), pronóstico $t+1$ y detección de alertas meteorológicas operativas |
| `results/` | 4–5 | Artefactos generados (modelos, métricas, predicciones, figuras y entorno) |
| `data/hour.csv` | 1 | Dataset UCI Bike Sharing (ID 275) |
| `assets/arquitectura_rnn.png` | 1 | Diagrama de la arquitectura RNN muchos a uno |

---

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
- **Parte 4 (del 4.7 al 5.4):**
  - Protocolo de validación por bloque temporal (sin mezcla aleatoria) y contexto solo pasado en las ventanas.
  - Reproducibilidad: versiones de bibliotecas, semilla `42`, dispositivo y comandos (`results/environment.json`).
  - Rejilla 3×3 entrenada: MAE, RMSE, MASE, parámetros, épocas y tiempo por configuración.
  - Curvas de pérdida, selección de la configuración y evaluación única en prueba frente a los baselines.
- **Parte 5 (del 5.5 al final):**
  - Desglose del error MAE por hora del día (0 a 23), horas punta matutina/vespertina y valles.
  - Comparativa de desempeño entre días laborables y fines de semana / feriados.
  - Evaluación sobre los días de máxima demanda diaria acumulada y sensibilidad ante lluvia/nieve (`weathersit`).
  - Demostración interactiva en tiempo real con selector de ventana histórica (24/48 h), pronóstico para $t+1$ y sistema de alertas meteorológicas operativas.
  - Discusión técnica de hallazgos, limitaciones operativas y conclusiones finales del proyecto.

### Artefactos de `results/`

| Archivo | Contenido |
|---|---|
| `grid_results.csv` | Métricas de validación, parámetros, épocas y tiempo de C1–C9 |
| `histories.json` | Pérdida MSE de entrenamiento y validación por época |
| `val_predictions.csv` | Predicciones de validación de las 9 configuraciones y de los baselines |
| `selection.json` | Configuración elegida (C6: $L=24, H=64$) y criterio |
| `models/C*.pt`, `best_model.pt` | Pesos (`state_dict`) y metadatos de cada configuración y del modelo final |
| `scaler.json` | Mínimos/máximos del escalador ajustado solo en entrenamiento |
| `test_predictions.csv` | Predicciones de prueba (RNN y baselines) con hora, calendario y clima |
| `test_metrics.csv` | MAE, RMSE, MASE y mejora vs. persistencia en prueba |
| `environment.json`, `train_config.json` | Versiones, dispositivo y configuración de entrenamiento |
| `figures/` | Figuras exportadas para el informe académico |
