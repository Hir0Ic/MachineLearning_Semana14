# Pronóstico horario de alquiler de bicicletas con una RNN simple · Grupo 03

App interactiva de Streamlit con análisis exploratorio de datos (EDA), preprocesamiento, análisis de fuga de información y arquitectura de una Red Neuronal Recurrente (RNN).

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

| Archivo | Contenido |
|---|---|
| `app.py` | Interfaz interactiva de Streamlit (Problema, EDA, Partición y Preprocesamiento, Paso RNN, Fuga, Arquitectura y Dataset) |
| `features.py` | Definición de $y = \text{cnt}(t+1)$, ingeniería de características cíclicas y exclusión de variables con fuga |
| `preprocessing.py` | Partición cronológica 70/15/15, escalador MinMax ajustado solo en Train y tensor de ventanas $(N, L, F)$ para $L \in \{12, 24, 48\}$ |
| `data.py` | Funciones de carga de `hour.csv` y compatibilidad de datos |
| `metrics.py` | Métricas de evaluación (MAE, RMSE, MASE), líneas base (persistencia y semanal) y cálculo de parámetros RNN |
| `config.py` | Semilla fija (`SEED = 42`), rutas, hiperparámetros y constantes |
| `data/hour.csv` | Dataset UCI Bike Sharing (ID 275) |
| `assets/arquitectura_rnn.png` | Diagrama de la arquitectura RNN muchos a uno |

## Secciones Cubiertas

- **Parte 1 (del 1 al 3.2):** Introducción, marco teórico, fuga empírica de información ($casual + registered = cnt$), arquitectura RNN, métricas y ficha del dataset.
- **Parte 2 (del 3.3 al 4.2):** 
  - $y = cnt(t+1)$ y selección de características finales.
  - EDA exhaustivo: patrones horarios, picos de hora punta, días laborales vs. fines de semana y estacionalidad.
  - Serie temporal completa coloreada con partición cronológica (Entrenamiento 70%, Validación 15%, Prueba 15%).
  - Escalado ajustado exclusivamente con entrenamiento.
  - Ventanas deslizantes $L \in \{12, 24, 48\}$ y dimensiones del tensor $(N, L, F)$.
