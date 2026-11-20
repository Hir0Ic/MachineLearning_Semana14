"""Streamlit · Pronóstico horario de alquiler de bicicletas con una RNN simple (Grupo 03)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from config import DIAGRAM_PATH, FEATURES, SEED, set_seeds
from data import (VARIABLES, add_features, apply_minmax, fit_minmax,
                  load_hour, make_windows, temporal_split)
from metrics import (mae, mae_naive_train, mase, persistence_forecast,
                     rmse, rnn_param_count, same_hour_mean_forecast)

st.set_page_config(page_title="RNN · Bike Sharing", page_icon="🚲", layout="wide")
set_seeds(SEED)

BLUE, ORANGE, GREEN, RED = "#2B579A", "#C8581A", "#2E7D4F", "#B5251D"


@st.cache_data
def get_data():
    df = add_features(load_hour())
    i_tr, i_va = temporal_split(len(df))
    return df, i_tr, i_va


df, I_TR, I_VA = get_data()

st.title("Pronóstico horario de alquiler de bicicletas con una RNN simple")
st.caption("Sistemas Inteligentes y Machine Learning · UPN · Grupo 03")

tabs = st.tabs([
    "Problema y objetivos",
    "Paso de la RNN y tensor",
    "Fuga de información",
    "Arquitectura y métricas",
    "Dataset y variables",
])

# ---------------------------------------------------------------- 1. Problema
with tabs[0]:
    st.header("Problema y objetivos")
    st.markdown(
        "Una empresa de bicicletas compartidas necesita que haya bicicletas disponibles "
        "donde y cuando se piden. La demanda cambia con la hora, el día de la semana, "
        "el clima y los feriados, así que una flota calculada con promedios queda corta "
        "en horas pico y sobra en horas valle."
    )
    st.subheader("Definición del problema")
    st.markdown(
        "Estimar la demanda total horaria de la **siguiente hora** a partir del historial "
        "reciente de demanda, el clima observado hasta la hora actual *t* y el calendario "
        "conocido de antemano."
    )
    st.latex(r"\hat y_{t+1} = f_\theta\big(x_{t-L+1}, \dots, x_t\big)")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Objetivo general")
        st.markdown("Entrenar una RNN simple para pronosticar los alquileres horarios y "
                    "verificar si mejora sobre líneas base temporales.")
    with c2:
        st.subheader("Objetivos específicos")
        st.markdown(
            "- Construir ventanas temporales sin fuga de observaciones futuras.\n"
            "- Explicar la recurrencia y el estado oculto.\n"
            "- Comparar tamaños de ventana y unidades recurrentes.\n"
            "- Analizar el error según hora del día y días de mayor demanda."
        )
    st.info("Alcance: horizonte t+1, demanda agregada de todo el sistema (Capital Bikeshare, "
            "Washington D. C.), archivo hour.csv de 2011 y 2012.")

# ---------------------------------------------------------- 2. Paso de la RNN
with tabs[1]:
    st.header("Un paso de la RNN y el tensor de una ventana")
    st.markdown(
        "Elige una ventana y mira cómo el estado oculto $h_t$ se actualiza hora a hora. "
        "Los pesos son **aleatorios y no entrenados** (solo ilustran el cálculo)."
    )
    c1, c2, c3, c4 = st.columns(4)
    L = c1.select_slider("Longitud de ventana L", [12, 24, 48], value=24)
    H = c2.select_slider("Unidades ocultas H", [16, 32, 64], value=32)
    B = c3.number_input("Tamaño de lote B", 1, 512, 32)
    t_idx = c4.slider("Hora t (índice)", L, len(df) - 2, 9000)

    # Escalado ajustado solo con entrenamiento
    scaler = fit_minmax(df.iloc[:I_TR])
    scaled = apply_minmax(df, scaler)
    win = scaled.iloc[t_idx - L + 1 : t_idx + 1][FEATURES].to_numpy()  # (L, F)
    F = len(FEATURES)

    st.markdown(
        f"**Tensor de entrada:** `(B, L, F) = ({B}, {L}, {F})` · una muestra es una matriz "
        f"`X ∈ ℝ^({L}×{F})`. **Objetivo:** cnt en {df.timestamp[t_idx + 1]:%Y-%m-%d %H:00} = "
        f"**{int(df.cnt[t_idx + 1])}** bicicletas."
    )
    show = df.iloc[t_idx - L + 1 : t_idx + 1][["timestamp"] + FEATURES].reset_index(drop=True)
    st.dataframe(show, height=260, use_container_width=True)

    rng = np.random.default_rng(SEED)
    k = 1 / np.sqrt(H)
    W = rng.uniform(-k, k, (H, F))
    U = rng.uniform(-k, k, (H, H))
    b = rng.uniform(-k, k, H)
    h = np.zeros(H)
    hs = []
    for x in win:
        h = np.tanh(W @ x + U @ h + b)
        hs.append(h.copy())
    hs = np.array(hs)  # (L, H)

    st.latex(r"h_t = \tanh(W x_t + U h_{t-1} + b), \qquad \hat y_{t+1} = V h_t + c")
    step = st.slider("Paso τ dentro de la ventana", 1, L, L)
    left, right = st.columns([3, 2])
    with left:
        fig, ax = plt.subplots(figsize=(7, 3.2))
        im = ax.imshow(hs.T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        ax.axvline(step - 1, color="black", lw=1.5)
        ax.set_xlabel("paso τ de la ventana")
        ax.set_ylabel("unidad oculta")
        ax.set_title("Estado oculto $h_τ$ en cada paso")
        fig.colorbar(im, ax=ax, label="activación (tanh)")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with right:
        prev = hs[step - 2] if step > 1 else np.zeros(H)
        pre = W @ win[step - 1] + U @ prev + b
        st.markdown(f"**Paso τ = {step}**")
        st.markdown(
            f"- `‖W·x‖` = {np.linalg.norm(W @ win[step - 1]):.3f}\n"
            f"- `‖U·h_(τ−1)‖` = {np.linalg.norm(U @ prev):.3f}"
            f"{'  (h₀ = 0)' if step == 1 else ''}\n"
            f"- `‖h_τ‖` = {np.linalg.norm(hs[step - 1]):.3f}\n"
            f"- Activaciones antes de tanh: de {pre.min():.2f} a {pre.max():.2f}; "
            f"después: de {hs[step - 1].min():.2f} a {hs[step - 1].max():.2f}"
        )
        st.caption("Los mismos W, U y b se reutilizan en los L pasos. Solo el último "
                   "estado h_t pasa a la capa lineal (modalidad muchos a uno).")

# ------------------------------------------------------------------- 3. Fuga
with tabs[2]:
    st.header("Fuga de información: casual + registered = cnt")
    ok = (df.casual + df.registered == df.cnt).mean()
    st.markdown(
        f"En el {ok:.0%} de las {len(df):,} filas se cumple exactamente "
        "`casual + registered = cnt`. Si los valores de la hora t+1 entran como "
        "predictores, el modelo reconstruye el objetivo con una suma."
    )
    st.dataframe(df[["timestamp", "casual", "registered", "cnt"]].head(8),
                 use_container_width=True, hide_index=True)

    st.subheader("Demostración con una regresión lineal")
    st.markdown("Se ajusta por mínimos cuadrados con el tramo de entrenamiento y se mide "
                "el error en el tramo de **prueba** (partición cronológica).")
    s = df.cnt.to_numpy(float)
    ca, rg = df.casual.to_numpy(float), df.registered.to_numpy(float)
    t = np.arange(24, len(df) - 1)
    y_next = s[t + 1]
    X_ok = np.column_stack([np.ones(len(t)), s[t], s[t - 1], s[t - 23],
                            df.temp.to_numpy()[t], df.hum.to_numpy()[t],
                            df.workingday.to_numpy()[t]])
    X_leak = np.column_stack([ca[t + 1], rg[t + 1]])
    tr = t < I_TR
    te = t >= I_VA

    def fit_eval(X):
        coef, *_ = np.linalg.lstsq(X[tr], y_next[tr], rcond=None)
        p = X[te] @ coef
        return mae(y_next[te], p), rmse(y_next[te], p)

    res = pd.DataFrame(
        {"MAE": [], "RMSE": []}, dtype=float
    )
    for name, X in [("Sin fuga: cnt(t), rezagos, clima(t), calendario", X_ok),
                    ("Con fuga: casual(t+1) y registered(t+1)", X_leak)]:
        m, r = fit_eval(X)
        res.loc[name] = [m, r]
    st.dataframe(res.style.format("{:.2f}"), use_container_width=True)
    st.warning("El error casi nulo del modelo con fuga **no es capacidad de pronóstico**: "
               "en uso real casual(t+1) y registered(t+1) no existen todavía.")
    st.markdown(
        "**Otras fugas a evitar:** usar temp, hum o windspeed de t+1 (equivale a tener un "
        "pronóstico meteorológico perfecto) y ajustar el escalado con datos de validación o "
        "prueba. El calendario de t+1 sí puede usarse porque se conoce con antelación."
    )

# ------------------------------------------------ 4. Arquitectura y métricas
with tabs[3]:
    st.header("Diagrama de arquitectura y métricas")
    st.image(str(DIAGRAM_PATH), caption="Figura 1. Arquitectura de la RNN utilizada",
             use_container_width=True)

    st.subheader("Parámetros entrenables")
    c1, c2, c3 = st.columns(3)
    Fp = c1.number_input("F (características)", 1, 64, len(FEATURES))
    Hp = c2.select_slider("H (unidades)", [16, 32, 64], value=32, key="hp")
    Lp = c3.select_slider("Capas recurrentes", [1, 2], value=1)
    rows = [
        ("Entrada X", "(B, L, F)", f"(B, L, {Fp})"),
        ("Pesos de entrada W", "(H, F)", f"({Hp}, {Fp})"),
        ("Pesos recurrentes U", "(H, H)", f"({Hp}, {Hp})"),
        ("Estado oculto h_t", "(B, H)", f"(B, {Hp})"),
        ("Cabeza lineal (V, c)", "(1, H) y (1)", f"(1, {Hp}) y (1)"),
        ("Salida ŷ", "(B, 1)", "(B, 1)"),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["Elemento", "Forma general", "Forma con tus valores"]),
                 hide_index=True, use_container_width=True)
    st.metric("Total de parámetros entrenables", f"{rnn_param_count(Fp, Hp, Lp):,}")

    st.subheader("Métricas")
    st.latex(r"\mathrm{MAE}=\frac1n\sum_i |y_i-\hat y_i|\qquad "
             r"\mathrm{RMSE}=\sqrt{\frac1n\sum_i (y_i-\hat y_i)^2}\qquad "
             r"\mathrm{MASE}=\frac{\mathrm{MAE}}{\mathrm{MAE}_{\text{ingenuo}}}")
    st.markdown(
        "- **MAE**: error medio en bicicletas por hora.\n"
        "- **RMSE**: mismas unidades; penaliza más los errores grandes.\n"
        "- **MASE**: divide entre el MAE de la persistencia calculado **solo en entrenamiento**; "
        "menor que 1 supera al ingenuo."
    )
    st.markdown("**Ejemplo con las dos líneas base** (partición 70/15/15 cronológica; "
                "las dos se evalúan sobre el tramo de prueba):")
    idx = np.arange(I_VA, len(df) - 1)
    y_true = s[idx + 1]
    base = {
        "A · Persistencia ŷ(t+1) = y(t)": persistence_forecast(s, idx),
        "B · Promedio de la misma hora, últimos 7 días": same_hour_mean_forecast(s, idx, 7),
    }
    tab = pd.DataFrame(
        {n: [mae(y_true, p), rmse(y_true, p), mase(y_true, p, s[:I_TR])] for n, p in base.items()},
        index=["MAE", "RMSE", "MASE"],
    ).T
    st.dataframe(tab.style.format("{:.2f}"), use_container_width=True)
    st.caption(f"MAE ingenuo en entrenamiento (denominador del MASE): {mae_naive_train(s[:I_TR]):.2f}. "
               "El MASE puede superar 1 aun para la persistencia porque la demanda de 2012 "
               "es mayor y más variable que la del tramo de entrenamiento.")

# --------------------------------------------------- 5. Dataset y variables
with tabs[4]:
    st.header("Ficha del dataset y variables")
    ficha = pd.DataFrame(
        [
            ("Nombre", "Bike Sharing Dataset"),
            ("Fuente oficial", "UCI Machine Learning Repository (ID 275)"),
            ("Enlace", "https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset"),
            ("Archivo utilizado", "hour.csv"),
            ("Período y granularidad", f"{df.timestamp.min():%Y-%m-%d} a {df.timestamp.max():%Y-%m-%d}, registros horarios"),
            ("Registros y columnas", f"{len(df):,} filas y {df.shape[1] - 3} columnas originales; "
                                      f"{int(df.isna().sum().sum())} valores faltantes; "
                                      f"{int(df.duplicated().sum())} duplicados"),
            ("Licencia", "Creative Commons Attribution 4.0 (CC BY 4.0)"),
            ("Autoría", "Hadi Fanaee-T (2013). DOI: 10.24432/C5W894"),
        ],
        columns=["Característica", "Detalle"],
    )
    st.dataframe(ficha, hide_index=True, use_container_width=True)

    st.subheader("Variables y su rol")
    st.dataframe(
        pd.DataFrame(VARIABLES, columns=["Variable", "Descripción", "Unidad / rango",
                                         "Rol en el trabajo", "¿Disponible antes de t+1?"]),
        hide_index=True, use_container_width=True,
    )
    st.subheader("Demanda promedio por hora del día")
    by_hr = df.groupby(["workingday", "hr"])["cnt"].mean().unstack(0)
    by_hr.columns = ["Fin de semana / feriado", "Día laborable"]
    st.line_chart(by_hr)
