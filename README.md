# Pronóstico horario de alquiler de bicicletas con una RNN simple · Grupo 03

App de Streamlit con 5 pestañas: problema y objetivos, paso de la RNN (h_t) y tensor de una
ventana, fuga de información (casual + registered = cnt), diagrama de arquitectura y métricas,
y ficha del dataset.

## Ejecutar
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Estructura
| Archivo | Contenido |
|---|---|
| `app.py` | Interfaz de Streamlit (5 pestañas) |
| `config.py` | Rutas, semilla (`SEED = 42`), partición 70/15/15, lista de características |
| `data.py` | Carga de `hour.csv`, características, escalado solo con entrenamiento, ventanas `(N, L, F)` |
| `metrics.py` | MAE, RMSE, MASE, líneas base y conteo de parámetros |
| `data/hour.csv` | UCI Bike Sharing (ID 275), CC BY 4.0, DOI 10.24432/C5W894 |
| `assets/arquitectura_rnn.png` | Figura 1 del informe |

## Notas
- Los pesos de la pestaña 2 son aleatorios (semilla fija) y solo ilustran el cálculo de h_t.
- Las características de `config.FEATURES` (F = 8) son una propuesta: ajústalas a la sección 3.3.
- Fuente: Fanaee-T, H. (2013). Bike Sharing [Dataset]. UCI Machine Learning Repository.
