"""Streamlit · Pronóstico horario de alquiler de bicicletas con una RNN simple (Grupo 03)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from baselines import evaluate_baselines
from config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_SEASONAL_DAYS,
    DIAGRAM_PATH,
    FEATURES,
    HIDDEN_SIZES,
    SEED,
    TARGET,
    TRAIN_FRAC,
    VAL_FRAC,
    WINDOW_SIZES,
    set_seeds,
)
from data import VARIABLES, load_hour
from features import EXCLUDED_LEAKAGE, EXCLUDED_REDUNDANT, add_cyclic_features
from metrics import (mae, mae_naive_train, mase, persistence_forecast,
                     rmse, rnn_param_count, same_hour_mean_forecast)
from model import (
    SimpleRNNForecaster,
    architecture_table,
    count_parameters,
    parameter_breakdown,
)
from preprocessing import (TemporalMinMaxScaler, create_sliding_windows,
                           split_dataframe, temporal_split_indices)
from train import TrainConfig, build_hyperparameter_grid, training_config_table
import evaluate as ev

st.set_page_config(page_title="RNN · Bike Sharing", page_icon="🚲", layout="wide")
set_seeds(SEED)

BLUE, ORANGE, GREEN, RED = "#2B579A", "#C8581A", "#2E7D4F", "#B5251D"


@st.cache_data
def get_dataset():
    raw_df = load_hour()
    full_df = add_cyclic_features(raw_df)
    i_tr, i_va = temporal_split_indices(len(full_df), TRAIN_FRAC, VAL_FRAC)
    return full_df, i_tr, i_va


df, I_TR, I_VA = get_dataset()

st.title("Pronóstico horario de alquiler de bicicletas con una RNN simple")
st.caption("Sistemas Inteligentes y Machine Learning · UPN · Grupo 03")

# Pestañas principales por integrante (Partes 1 a 4)
tabs = st.tabs([
    "📘 PARTE 1: Problema, Fundamentos y Datos",
    "📊 PARTE 2: Entradas, EDA y Preprocesamiento",
    "🧠 PARTE 3: Baselines y Arquitectura RNN",
    "🧪 PARTE 4: Experimentos y Resultados",
])

# ====================================================================================
# TAB 1: PARTE 1
# ====================================================================================
with tabs[0]:
    st.header("PARTE 1: Problema, Marco Teórico, Fundamentos y Datos")
    
    # ------------------------------------------------------------- 1. Problema y objetivos
    st.subheader("1. Introducción, Problema y Objetivos")
    st.markdown(
        "Una empresa de bicicletas compartidas necesita que haya unidades disponibles "
        "donde y cuando se solicitan. La demanda varía continuamente según la hora, el día de la semana, "
        "el clima y los días festivos; un dimensionamiento estático o basado en promedios simples genera "
        "déficit en horas punta y exceso de unidades ociosas en horas valle."
    )
    st.markdown(
        "**Definición del problema:** Estimar la demanda total horaria de la **siguiente hora** ($t+1$) "
        "a partir del historial reciente de demanda, las variables meteorológicas observadas hasta la hora $t$ "
        "y el calendario conocido de antemano."
    )
    st.latex(r"\hat y_{t+1} = f_\theta\big(x_{t-L+1}, \dots, x_t\big)")
    
    col_obj1, col_obj2 = st.columns(2)
    with col_obj1:
        st.markdown("**Objetivo General:**")
        st.markdown("Entrenar y evaluar una Red Neuronal Recurrente (RNN) simple para pronosticar los alquileres "
                    "horarios y comprobar rigurosamente si supera a las líneas base temporales estándar.")
    with col_obj2:
        st.markdown("**Objetivos Específicos:**")
        st.markdown(
            "- Construir secuencias temporales en ventanas deslizantes sin fuga de datos futuros.\n"
            "- Modelar la recurrencia mediante la actualización del estado oculto $h_t$.\n"
            "- Evaluar el impacto de la longitud de ventana ($L \\in \\{12, 24, 48\\}$) y unidades ocultas.\n"
            "- Analizar el error en tramos pico vs. valles usando métricas apropiadas (MAE, RMSE, MASE)."
        )
    st.info("📌 **Alcance del estudio:** Predicción a horizonte $t+1$, demanda agregada del sistema "
            "Capital Bikeshare (Washington D. C.), registros horarios 2011-2012 (`hour.csv`).")

    st.divider()

    # ------------------------------------------------------------- 2. Paso de la RNN
    st.subheader("2. Marco Teórico: Un paso de la RNN y el tensor de una ventana")
    st.markdown(
        "La RNN mantiene un vector de memoria o estado oculto $h_t$ que se actualiza paso a paso. "
        "A continuación puedes explorar el tensor de entrada y la dinámica del estado oculto. "
        "*(Los pesos presentados son ilustrativos con semilla fija para visualizar el cálculo).* "
    )
    
    c1, c2, c3, c4 = st.columns(4)
    L = c1.select_slider("Longitud de ventana L", [12, 24, 48], value=24, key="p1_L")
    H = c2.select_slider("Unidades ocultas H", [16, 32, 64], value=32, key="p1_H")
    B = c3.number_input("Tamaño de lote B", 1, 512, 32, key="p1_B")
    t_idx = c4.slider("Hora t (índice en serie)", L, len(df) - 2, 9000, key="p1_t")

    scaler_demo = TemporalMinMaxScaler(cols=FEATURES).fit(df.iloc[:I_TR])
    scaled = scaler_demo.transform(df)
    win = scaled.iloc[t_idx - L + 1 : t_idx + 1][FEATURES].to_numpy()
    F = len(FEATURES)

    st.markdown(
        f"**Dimensiones del tensor:** `(B, L, F) = ({B}, {L}, {F})`. Cada muestra individual es una matriz "
        f"$X_i \\in \\mathbb{{R}}^{{{L} \\times {F}}}$. **Objetivo:** $\\text{{cnt}}$ en "
        f"{df.timestamp[t_idx + 1]:%Y-%m-%d %H:00} = **{int(df.cnt[t_idx + 1])}** bicicletas."
    )
    show = df.iloc[t_idx - L + 1 : t_idx + 1][["timestamp"] + FEATURES].reset_index(drop=True)
    st.dataframe(show, height=220, use_container_width=True)

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
    hs = np.array(hs)

    st.latex(r"h_t = \tanh(W x_t + U h_{t-1} + b), \qquad \hat y_{t+1} = V h_t + c")
    step = st.slider("Paso τ dentro de la ventana", 1, L, L, key="p1_step")
    left, right = st.columns([3, 2])
    with left:
        fig, ax = plt.subplots(figsize=(7, 3.2))
        im = ax.imshow(hs.T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        ax.axvline(step - 1, color="black", lw=1.5)
        ax.set_xlabel("Paso τ de la ventana")
        ax.set_ylabel("Unidad oculta")
        ax.set_title("Activación del estado oculto $h_τ$ a través del tiempo")
        fig.colorbar(im, ax=ax, label="tanh")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with right:
        prev = hs[step - 2] if step > 1 else np.zeros(H)
        pre = W @ win[step - 1] + U @ prev + b
        st.markdown(f"**Detalle del paso τ = {step}**")
        st.markdown(
            f"- `‖W·x‖` = {np.linalg.norm(W @ win[step - 1]):.3f}\n"
            f"- `‖U·h_(τ−1)‖` = {np.linalg.norm(U @ prev):.3f}"
            f"{'  (h₀ = 0)' if step == 1 else ''}\n"
            f"- `‖h_τ‖` = {np.linalg.norm(hs[step - 1]):.3f}\n"
            f"- Activación pre-tanh: [{pre.min():.2f}, {pre.max():.2f}]\n"
            f"- Activación post-tanh: [{hs[step - 1].min():.2f}, {hs[step - 1].max():.2f}]"
        )
        st.caption("Los parámetros W, U y b se comparten en todos los pasos L. Solo el último "
                   "estado oculto h_L se conecta a la cabeza de salida lineal (Many-to-One).")

    st.divider()

    # ------------------------------------------------------------- Fuga de información
    st.subheader("Fuga de Información: casual + registered = cnt")
    ok = (df.casual + df.registered == df.cnt).mean()
    st.markdown(
        f"En el **{ok:.0%}** de las {len(df):,} filas del dataset se cumple la identidad matemática exacta: "
        "`casual + registered = cnt`. Si las observaciones de $t+1$ de estas variables se incluyeran en el modelo, "
        "éste aprendería una suma trivial y no la dinámica temporal real."
    )
    st.dataframe(df[["timestamp", "casual", "registered", "cnt"]].head(6),
                 use_container_width=True, hide_index=True)

    st.markdown("**Demostración empírica mediante Mínimos Cuadrados Ordinarios (tramo de prueba):**")
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

    res = pd.DataFrame({"MAE": [], "RMSE": []}, dtype=float)
    for name, X in [("Sin fuga: cnt(t), rezagos, clima(t), calendario", X_ok),
                    ("Con fuga directa: casual(t+1) y registered(t+1)", X_leak)]:
        m, r = fit_eval(X)
        res.loc[name] = [m, r]
    st.dataframe(res.style.format("{:.2f}"), use_container_width=True)
    st.warning("⚠️ El error casi nulo del modelo con fuga no refleja capacidad predictiva: "
               "en producción ni `casual(t+1)` ni `registered(t+1)` están disponibles antes de la hora t+1.")

    st.divider()

    # ------------------------------------------------------------- Arquitectura y Métricas
    st.subheader("Diagrama de Arquitectura y Métricas")
    col_diag, col_params = st.columns([1, 1])
    with col_diag:
        st.image(str(DIAGRAM_PATH), caption="Figura 1. Arquitectura RNN Many-to-One", use_container_width=True)
    with col_params:
        st.markdown("**Cálculo de parámetros entrenables:**")
        Fp = st.number_input("Características de entrada (F)", 1, 64, len(FEATURES), key="p1_Fp")
        Hp = st.select_slider("Unidades ocultas (H)", [16, 32, 64], value=32, key="p1_Hp")
        Lp = st.select_slider("Capas recurrentes", [1, 2], value=1, key="p1_Lp")
        rows = [
            ("Entrada X", "(B, L, F)", f"(B, L, {Fp})"),
            ("Pesos entrada W", "(H, F)", f"({Hp}, {Fp})"),
            ("Pesos recurrentes U", "(H, H)", f"({Hp}, {Hp})"),
            ("Estado oculto h_t", "(B, H)", f"(B, {Hp})"),
            ("Cabeza lineal (V, c)", "(1, H) y (1)", f"(1, {Hp}) y (1)"),
            ("Salida ŷ", "(B, 1)", "(B, 1)"),
        ]
        st.dataframe(pd.DataFrame(rows, columns=["Elemento", "Fórmula", "Configuración actual"]),
                     hide_index=True, use_container_width=True)
        st.metric("Total de parámetros de la red", f"{rnn_param_count(Fp, Hp, Lp):,}")

    st.markdown("**Métricas de evaluación seleccionadas:**")
    st.latex(r"\mathrm{MAE}=\frac1n\sum_i |y_i-\hat y_i|\qquad \mathrm{RMSE}=\sqrt{\frac1n\sum_i (y_i-\hat y_i)^2}\qquad \mathrm{MASE}=\frac{\mathrm{MAE}}{\mathrm{MAE}_{\text{ingenuo}}}")
    st.markdown("Comparación con líneas base en el tramo de **Prueba (15%)**:")
    idx = np.arange(I_VA, len(df) - 1)
    y_true = s[idx + 1]
    base = {
        "A · Persistencia: ŷ(t+1) = y(t)": persistence_forecast(s, idx),
        "B · Promedio estacional: misma hora últimos 7 días": same_hour_mean_forecast(s, idx, 7),
    }
    tab = pd.DataFrame(
        {n: [mae(y_true, p), rmse(y_true, p), mase(y_true, p, s[:I_TR])] for n, p in base.items()},
        index=["MAE", "RMSE", "MASE"],
    ).T
    st.dataframe(tab.style.format("{:.2f}"), use_container_width=True)

    st.divider()

    # ------------------------------------------------------------- 3.1 y 3.2 Dataset y Variables
    st.subheader("3.1 Fuente y Trazabilidad & 3.2 Variables")
    ficha = pd.DataFrame(
        [
            ("Nombre del Dataset", "Bike Sharing Dataset"),
            ("Repositorio", "UCI Machine Learning Repository (ID 275)"),
            ("URL de procedencia", "https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset"),
            ("Archivo procesado", "hour.csv"),
            ("Rango temporal", f"{df.timestamp.min():%Y-%m-%d} a {df.timestamp.max():%Y-%m-%d} (Registros horarios)"),
            ("Total de observaciones", f"{len(df):,} filas | {int(df.isna().sum().sum())} nulos | {int(df.duplicated().sum())} duplicados"),
            ("Licencia", "Creative Commons Attribution 4.0 (CC BY 4.0)"),
            ("Cita académica", "Hadi Fanaee-T (2013). DOI: 10.24432/C5W894"),
        ],
        columns=["Atributo", "Detalle"]
    )
    st.dataframe(ficha, hide_index=True, use_container_width=True)

    st.markdown("**Tabla de Variables y su Rol:**")
    st.dataframe(
        pd.DataFrame(VARIABLES, columns=["Variable", "Descripción", "Unidad / rango", "Rol en el trabajo", "¿Disponible antes de t+1?"]),
        hide_index=True, use_container_width=True,
    )


# ====================================================================================
# TAB 2: PARTE 2
# ====================================================================================
with tabs[1]:
    st.header("PARTE 2: Entradas, EDA y Preprocesamiento")

    # ------------------------------------------------------------- 3.3 Entradas y salidas
    st.subheader("3.3 Formulación de Entradas y Salidas")
    col_feat1, col_feat2 = st.columns([3, 2])
    with col_feat1:
        st.markdown("**Características seleccionadas ($F=8$):**")
        st.dataframe(pd.DataFrame({
            "Característica": FEATURES,
            "Tipo": [
                "Demanda histórica (rezago)",
                "Climática (normalizada)",
                "Climática (normalizada)",
                "Climática (normalizada)",
                "Cíclica (hora seno)",
                "Cíclica (hora coseno)",
                "Calendario (binaria)",
                "Calendario (binaria)"
            ],
            "Disponibilidad": ["Hasta t", "Hasta t", "Hasta t", "Hasta t", "Conocida", "Conocida", "Conocida", "Conocida"]
        }), hide_index=True, use_container_width=True)
    with col_feat2:
        st.markdown("**Variable objetivo ($y$):**")
        st.latex(r"y = \text{cnt}(t+1)")
        st.markdown(
            "Alquileres totales registrados en la siguiente hora inmediata. "
            "Permite a los operadores anticipar movimientos de vehículos con 1 hora de ventana."
        )
        st.info(f"**Variables excluidas por fuga directa:** `{', '.join(EXCLUDED_LEAKAGE)}`\n\n"
                f"**Variables excluidas por redundancia:** `{', '.join(EXCLUDED_REDUNDANT)}`")

    st.divider()

    # ------------------------------------------------------------- 3.4 EDA
    st.subheader("3.4 Análisis Exploratorio de Datos (EDA: hora, día, estación y picos)")
    eda_view = st.radio(
        "Selecciona la vista del análisis exploratorio:",
        ["1. Patrón Horario y Picos (Día Laboral vs. Fin de Semana)", "2. Demanda por Estación Climática", "3. Influencia de Temperatura y Clima"],
        horizontal=True
    )

    if eda_view == "1. Patrón Horario y Picos (Día Laboral vs. Fin de Semana)":
        by_hr_wd = df.groupby(["workingday", "hr"])["cnt"].mean().unstack(0)
        by_hr_wd.columns = ["Fin de semana / Feriado", "Día laborable"]
        
        fig, ax = plt.subplots(figsize=(10, 4.2))
        ax.plot(by_hr_wd.index, by_hr_wd["Día laborable"], marker="o", color=BLUE, lw=2.2, label="Día laborable (Picos 08:00 y 17:00-18:00)")
        ax.plot(by_hr_wd.index, by_hr_wd["Fin de semana / Feriado"], marker="s", color=ORANGE, lw=2.2, label="Fin de semana / Feriado (Pico 12:00-16:00)")
        ax.set_xticks(range(0, 24))
        ax.set_xlabel("Hora del día (0-23)")
        ax.set_ylabel("Demanda media (bicicletas/hora)")
        ax.set_title("Patrón Horario de Demanda: Bimodal (Laboral) vs. Unimodal (Fines de semana)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
        st.info("💡 **Conclusión del EDA:** Los días laborables están dominados por desplazamientos laborales/académicos (hora punta mañana y tarde), mientras que fines de semana muestran actividad recreativa en las horas centrales del día.")

    elif eda_view == "2. Demanda por Estación Climática":
        season_map = {1: "Invierno", 2: "Primavera", 3: "Verano", 4: "Otoño"}
        df_season = df.copy()
        df_season["season_name"] = df_season["season"].map(season_map)
        season_means = df_season.groupby("season_name")["cnt"].mean().reindex(["Invierno", "Primavera", "Verano", "Otoño"])
        
        fig, ax = plt.subplots(figsize=(8, 3.8))
        bars = ax.bar(season_means.index, season_means.values, color=["#4a90e2", "#50e3c2", "#f5a623", "#d0021b"], alpha=0.85)
        ax.set_ylabel("Demanda media (cnt)")
        ax.set_title("Demanda media de alquileres según la estación")
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 3, f"{yval:.1f}", ha="center", va="bottom", fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        st.pyplot(fig)
        plt.close(fig)
        st.caption("El verano y el otoño concentran las mayores demandas promedio de alquileres.")

    else:
        fig, ax = plt.subplots(figsize=(9, 4))
        scatter = ax.scatter(df["temp"], df["cnt"], alpha=0.15, c=df["cnt"], cmap="viridis", s=10)
        ax.set_xlabel("Temperatura normalizada (temp)")
        ax.set_ylabel("Alquileres por hora (cnt)")
        ax.set_title("Relación entre Temperatura y Demanda Horaria")
        fig.colorbar(scatter, ax=ax, label="Demanda (cnt)")
        st.pyplot(fig)
        plt.close(fig)

    st.divider()

    # ------------------------------------------------------------- 4.1 Separación experimental
    st.subheader("4.1 Separación Experimental: Serie con Train, Val y Test coloreados")
    st.markdown(
        "Para preservar la dependencia temporal y evitar cualquier fuga retrospectiva, "
        "la partición se realiza estrictamente en **orden cronológico continuo** (sin barajar):"
    )

    train_df, val_df, test_df = split_dataframe(df)

    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.metric("1. Entrenamiento (70%)", f"{len(train_df):,} horas", f"{train_df.timestamp.min():%Y-%m-%d} a {train_df.timestamp.max():%Y-%m-%d}")
    col_s2.metric("2. Validación (15%)", f"{len(val_df):,} horas", f"{val_df.timestamp.min():%Y-%m-%d} a {val_df.timestamp.max():%Y-%m-%d}")
    col_s3.metric("3. Prueba (15%)", f"{len(test_df):,} horas", f"{test_df.timestamp.min():%Y-%m-%d} a {test_df.timestamp.max():%Y-%m-%d}")

    fig_split, ax_split = plt.subplots(figsize=(12, 4.2))
    ax_split.plot(train_df["timestamp"], train_df["cnt"], label="Entrenamiento (70%)", color=BLUE, lw=0.8, alpha=0.85)
    ax_split.plot(val_df["timestamp"], val_df["cnt"], label="Validación (15%)", color=ORANGE, lw=0.8, alpha=0.85)
    ax_split.plot(test_df["timestamp"], test_df["cnt"], label="Prueba (15%)", color=RED, lw=0.8, alpha=0.85)
    ax_split.axvline(train_df["timestamp"].max(), color="black", linestyle="--", alpha=0.7)
    ax_split.axvline(val_df["timestamp"].max(), color="black", linestyle="--", alpha=0.7)
    ax_split.set_xlabel("Línea temporal")
    ax_split.set_ylabel("Demanda horaria (cnt)")
    ax_split.set_title("Serie temporal completa: Partición cronológica Train (70%) / Val (15%) / Test (15%)")
    ax_split.legend(loc="upper left")
    ax_split.grid(True, linestyle=":", alpha=0.4)
    st.pyplot(fig_split)
    plt.close(fig_split)

    st.divider()

    # ------------------------------------------------------------- 4.2 Preprocesamiento
    st.subheader("4.2 Preprocesamiento: Escalado ajustado solo con Train y Tensores de Ventanas")
    
    st.markdown("**1. Escalado MinMax sin fuga:**")
    st.markdown("Los rangos $[min, max]$ de cada característica se calculan **únicamente con el tramo de entrenamiento**. Las transformaciones a validación y prueba usan estos mismos parámetros fijos:")
    scaler = TemporalMinMaxScaler(cols=FEATURES)
    scaler.fit(train_df)
    train_scaled = scaler.transform(train_df)
    
    stats_df = pd.DataFrame(scaler.stats_, index=["Mínimo (Train)", "Máximo (Train)"]).T
    st.dataframe(stats_df.style.format("{:.3f}"), use_container_width=True)

    st.markdown("**2. Ventanas deslizantes L = 12, 24, 48 y forma del tensor (N, L, F):**")
    selected_L = st.select_slider("Selecciona la longitud de ventana L a inspeccionar:", [12, 24, 48], value=24, key="p2_L")
    
    X_train_win, y_train_win = create_sliding_windows(train_scaled, train_df, L=selected_L)
    
    col_win1, col_win2, col_win3 = st.columns(3)
    col_win1.metric("Forma del tensor X (N, L, F)", f"({X_train_win.shape[0]:,}, {X_train_win.shape[1]}, {X_train_win.shape[2]})")
    col_win2.metric("Forma del vector y (N,)", f"({y_train_win.shape[0]:,},)")
    col_win3.metric("Tamaño en memoria (tensor X)", f"{X_train_win.nbytes / (1024 * 1024):.2f} MB")

    st.markdown(f"**Ejemplo de los primeros 6 pasos de la primera ventana $X_0$ ($L={selected_L}, F=8$):**")
    sample_win = pd.DataFrame(X_train_win[0], columns=FEATURES)
    st.dataframe(sample_win.head(6), use_container_width=True)
    st.caption(f"Salida correspondiente a predecir $y_0 = \\text{{cnt}}(t+1)$: **{int(y_train_win[0])}** bicicletas.")


# ====================================================================================
# TAB 3: PARTE 3
# ====================================================================================
with tabs[2]:
    st.header("PARTE 3: Modelos de Referencia (Baselines) y Diseño de la RNN")

    # ------------------------------------------------------------- 4.3 Baselines en validación
    st.subheader("4.3 Modelos de Referencia: Persistencia y Promedio Estacional en Validación")
    st.markdown(
        "Antes de entrenar la red recurrente se establecen **líneas base estrictamente causales** "
        "(usan solo observaciones $\\le t$) evaluadas sobre el **conjunto de validación (15 %, "
        f"{I_VA - I_TR:,} horas)**. El conjunto de prueba permanece reservado y no se consulta en esta etapa."
    )

    col_b1, col_b2 = st.columns([2, 1])
    seasonal_k = col_b1.select_slider(
        "Días recientes (k) para el promedio estacional (misma hora):",
        options=[1, 3, 7, 14],
        value=DEFAULT_SEASONAL_DAYS,
        key="p3_k",
        help="Con k=1 corresponde a la misma hora del día anterior: cnt(t+1-24). Con k=7 promedia los últimos 7 días.",
    )
    show_ridge = col_b2.checkbox(
        "Incluir Baseline C opcional (Regresión Ridge sobre rezagos)",
        value=True,
        key="p3_ridge",
    )

    base_table, base_preds, y_val_true, val_ts = evaluate_baselines(
        df,
        split="val",
        seasonal_days=seasonal_k,
        include_lag_reg=show_ridge,
        train_frac=TRAIN_FRAC,
        val_frac=VAL_FRAC,
    )

    st.dataframe(
        base_table.style.format(
            {
                "MAE": "{:.2f}",
                "RMSE": "{:.2f}",
                "MASE": "{:.3f}",
                "Mejora MAE vs. Persistencia (%)": "{:+.2f} %",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        f"MASE calculado respecto al MAE ingenuo de persistencia en entrenamiento "
        f"(`MAE_ingenuo = {mae_naive_train(df['cnt'].iloc[:I_TR]):.2f}` bicicletas/hora). "
        "Un MASE < 1 indica mejora sobre la persistencia de entrenamiento."
    )

    # Gráfica comparativa en una semana del tramo de validación
    st.markdown("**Comparación visual en una semana del tramo de validación (168 horas):**")
    max_start = max(0, len(y_val_true) - 168)
    week_start = st.slider(
        "Inicio de la ventana semanal dentro de validación (hora relativa):",
        min_value=0,
        max_value=max_start,
        value=min(240, max_start),
        step=24,
        key="p3_week",
    )
    sl = slice(week_start, week_start + 168)

    fig_b, ax_b = plt.subplots(figsize=(12, 4.0))
    ax_b.plot(val_ts.iloc[sl], y_val_true[sl], label="Real cnt(t+1)", color="#1F2A44", lw=2.0)
    ax_b.plot(
        val_ts.iloc[sl],
        base_preds["A. Persistencia: cnt(t)"][sl],
        label="A. Persistencia: cnt(t)",
        color=ORANGE,
        lw=1.3,
        linestyle="--",
        alpha=0.85,
    )
    seasonal_key = f"B. Promedio estacional (k={seasonal_k} días)"
    ax_b.plot(
        val_ts.iloc[sl],
        base_preds[seasonal_key][sl],
        label=seasonal_key,
        color=GREEN,
        lw=1.5,
        alpha=0.9,
    )
    if show_ridge and "C. Regresión Ridge sobre rezagos (opcional)" in base_preds:
        ax_b.plot(
            val_ts.iloc[sl],
            base_preds["C. Regresión Ridge sobre rezagos (opcional)"][sl],
            label="C. Regresión Ridge (rezagos)",
            color=BLUE,
            lw=1.4,
            linestyle="-.",
            alpha=0.85,
        )
    ax_b.set_xlabel("Fecha y hora (validación)")
    ax_b.set_ylabel("Alquileres horarios (cnt)")
    ax_b.set_title("Comportamiento de las líneas base en validación (tramo de 7 días)")
    ax_b.legend(loc="upper right")
    ax_b.grid(True, linestyle=":", alpha=0.5)
    fig_b.autofmt_xdate()
    fig_b.tight_layout()
    st.pyplot(fig_b)
    plt.close(fig_b)

    st.info(
        "💡 **Observación operativa:** La persistencia arrastra un desfase de 1 hora en subidas y bajadas "
        "bruscas (picos de 08:00 y 17:00), mientras que el promedio estacional captura el ciclo diario de "
        "24 horas pero suaviza variaciones por clima o cambio de día laborable a fin de semana."
    )

    st.divider()

    # ------------------------------------------------------------- 4.4 Diseño de la arquitectura
    st.subheader("4.4 Diseño de la Arquitectura: RNN con Selector de H y Conteo de Parámetros")
    st.markdown(
        "A partir del esquema general de la Sección 2.5, aquí se concretan los tensores y el número exacto "
        "de parámetros entrenables instanciando el módulo real `SimpleRNNForecaster` (`model.py`) en PyTorch."
    )

    c_a1, c_a2, c_a3, c_a4 = st.columns(4)
    p3_L = c_a1.select_slider("Ventana L (horas)", WINDOW_SIZES, value=24, key="p3_L")
    p3_H = c_a2.select_slider("Unidades ocultas H", HIDDEN_SIZES, value=32, key="p3_H")
    p3_layers = c_a3.select_slider("Capas recurrentes", [1, 2], value=1, key="p3_layers")
    p3_B = c_a4.select_slider("Batch B (ejemplo)", [32, 64, 128], value=DEFAULT_BATCH_SIZE, key="p3_B")

    F_dim = len(FEATURES)
    rnn_net = SimpleRNNForecaster(input_size=F_dim, hidden_size=p3_H, num_layers=p3_layers)
    n_params_torch = count_parameters(rnn_net)
    n_params_formula = rnn_param_count(F_dim, p3_H, p3_layers)
    with torch.no_grad():
        dummy_out = rnn_net(torch.zeros((p3_B, p3_L, F_dim), dtype=torch.float32))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tensor de entrada X", f"({p3_B}, {p3_L}, {F_dim})")
    m2.metric("Estado final h_t", f"({p3_B}, {p3_H})")
    m3.metric("Tensor de salida ŷ", str(tuple(dummy_out.shape)))
    m4.metric(
        "Parámetros entrenables",
        f"{n_params_torch:,}",
        "Coincide con fórmula" if n_params_torch == n_params_formula else "Revisar",
    )

    col_t7, col_bar = st.columns([3, 2])
    with col_t7:
        st.markdown("**Tabla 7 · Capas concretas de la arquitectura RNN (Many-to-One):**")
        arch_df = architecture_table(
            input_size=F_dim,
            seq_len=p3_L,
            hidden_size=p3_H,
            num_layers=p3_layers,
            batch_size=p3_B,
        )
        st.dataframe(arch_df, hide_index=True, use_container_width=True)
        with st.expander("Ver desglose por tensor de pesos y sesgos en PyTorch"):
            st.dataframe(
                pd.DataFrame(parameter_breakdown(F_dim, p3_H, p3_layers)),
                hide_index=True,
                use_container_width=True,
            )

    with col_bar:
        st.markdown("**Crecimiento cuadrático de parámetros ($\\mathcal{O}(H^2)$):**")
        p_1layer = [rnn_param_count(F_dim, h, 1) for h in HIDDEN_SIZES]
        p_2layer = [rnn_param_count(F_dim, h, 2) for h in HIDDEN_SIZES]
        x_pos = np.arange(len(HIDDEN_SIZES))
        width = 0.35
        fig_p, ax_p = plt.subplots(figsize=(5.5, 3.6))
        b1 = ax_p.bar(x_pos - width / 2, p_1layer, width, label="1 capa recurrente", color=BLUE)
        b2 = ax_p.bar(x_pos + width / 2, p_2layer, width, label="2 capas recurrentes", color=ORANGE)
        ax_p.set_xticks(x_pos)
        ax_p.set_xticklabels([f"H = {h}" for h in HIDDEN_SIZES])
        ax_p.set_ylabel("Parámetros entrenables")
        ax_p.set_title(f"Complejidad del modelo (F = {F_dim})")
        ax_p.legend()
        ax_p.grid(axis="y", linestyle="--", alpha=0.5)
        for bars in (b1, b2):
            for bar in bars:
                val = int(bar.get_height())
                ax_p.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + 120,
                    f"{val:,}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        fig_p.tight_layout()
        st.pyplot(fig_p)
        plt.close(fig_p)
        st.caption(
            "Los parámetros dependen de F y H, no de L: la recurrencia reutiliza (W, U, b) en todos los pasos."
        )

    st.divider()

    # ------------------------------------------------------------- 4.5 Configuración de aprendizaje
    st.subheader("4.5 Configuración de Aprendizaje: MSE, Adam, Batch, Clipping y Semilla")
    cfg_default = TrainConfig()
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Pérdida", "MSE")
    k2.metric("Optimizador", f"Adam (lr={cfg_default.lr:g})")
    k3.metric("Batch size", f"{cfg_default.batch_size}")
    k4.metric("Gradient clipping", f"L2 ≤ {cfg_default.grad_clip}")
    k5.metric("Semilla fija", f"SEED = {cfg_default.seed}")

    st.dataframe(training_config_table(cfg_default), hide_index=True, use_container_width=True)
    st.caption(
        "El bucle reproducible está implementado en `train.py` (`train_rnn`). Según la división del equipo, "
        "la Parte 4 (`experiments.py`) ejecuta el entrenamiento fuera de la app y almacena en `results/` "
        "el mejor modelo, el escalador ajustado y las predicciones."
    )

    st.divider()

    # ------------------------------------------------------------- 4.6 Plan de hiperparámetros
    st.subheader("4.6 Plan de Hiperparámetros: Rejilla 3×3 de Combinaciones a Probar")
    st.markdown(
        "Se evalúa la rejilla completa de **9 combinaciones** ($L \\in \\{12, 24, 48\\}$ horas $\\times$ "
        "$H \\in \\{16, 32, 64\\}$ unidades ocultas) manteniendo constantes: los datos ($F=8$), la partición "
        "cronológica (70/15/15), el escalador ajustado solo en entrenamiento, la configuración de la Tabla 8 "
        "y el criterio de selección (menor **MAE / RMSE en validación**)."
    )

    grid_layers = st.radio(
        "Capas recurrentes para proyectar la rejilla:",
        options=[1, 2],
        horizontal=True,
        index=0,
        key="p3_grid_layers",
    )
    grid_df = build_hyperparameter_grid(
        window_sizes=WINDOW_SIZES,
        hidden_sizes=HIDDEN_SIZES,
        input_size=F_dim,
        num_layers=grid_layers,
        n_train=I_TR,
    )
    st.dataframe(grid_df, hide_index=True, use_container_width=True)

    # Matriz visual 3x3 de combinaciones (L x H) en formato compacto centrado
    pivot_params = grid_df.pivot(index="Ventana L (h)", columns="Hidden H", values="Parámetros")
    pivot_ids = grid_df.pivot(index="Ventana L (h)", columns="Hidden H", values="ID")
    fig_g, ax_g = plt.subplots(figsize=(5.0, 2.5))
    im_g = ax_g.imshow(pivot_params.to_numpy(), cmap="Blues", aspect="auto")
    ax_g.set_xticks(range(len(pivot_params.columns)))
    ax_g.set_xticklabels([f"H = {h}" for h in pivot_params.columns], fontsize=9)
    ax_g.set_yticks(range(len(pivot_params.index)))
    ax_g.set_yticklabels([f"L = {l} h" for l in pivot_params.index], fontsize=9)
    ax_g.set_title("Mapa de la rejilla 3×3 (ID y parámetros entrenables)", fontsize=10)
    max_p = pivot_params.to_numpy().max()
    for r_idx in range(pivot_params.shape[0]):
        for c_idx in range(pivot_params.shape[1]):
            val = int(pivot_params.iloc[r_idx, c_idx])
            cid = pivot_ids.iloc[r_idx, c_idx]
            color_txt = "white" if val > max_p * 0.6 else "#1F2A44"
            ax_g.text(
                c_idx,
                r_idx,
                f"{cid}\n{val:,} params",
                ha="center",
                va="center",
                color=color_txt,
                fontweight="bold",
                fontsize=8.5,
            )
    cbar = fig_g.colorbar(im_g, ax=ax_g, label="Parámetros")
    cbar.ax.tick_params(labelsize=8)
    fig_g.tight_layout()
    _, col_map, _ = st.columns([1.2, 2.0, 1.2])
    with col_map:
        st.pyplot(fig_g, use_container_width=True)
    plt.close(fig_g)

    st.success(
        "✅ **Criterios fijos para la Parte 4:** Todas las configuraciones C1–C9 se comparan sobre las mismas "
        f"{I_VA - I_TR:,} horas de validación usando `train.py` con semilla `{SEED}`. Las columnas de tiempo, "
        "MAE val. y RMSE val. se completan en la ejecución experimental de la Parte 4 (`experiments.py`)."
    )


# ====================================================================================
# TAB 4: PARTE 4
# ====================================================================================
@st.cache_data
def get_results(_stamp):
    """Lee results/ (la clave `_stamp` cambia cuando se regeneran los archivos)."""
    return ev.load_results()


def _results_stamp():
    files = list(ev.RESULTS_DIR.glob("*.csv")) + list(ev.RESULTS_DIR.glob("*.json"))
    return tuple(sorted((f.name, f.stat().st_mtime) for f in files))


def _show(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _howto(what, how):
    """Recuadro 'Cómo leer esta sección' para la exposición."""
    st.info(f"**🔎 Qué muestra:** {what}\n\n**📖 Cómo leerlo:** {how}")


def _key(msg):
    st.success(f"**✅ Mensaje clave:** {msg}")


with tabs[3]:
    st.header("PARTE 4: Experimentos, Selección y Evaluación Final")

    status = ev.results_available()
    if not status["grid"]:
        st.warning("Aún no existen resultados en `results/`. Ejecuta desde la carpeta del proyecto:")
        st.code("python experiments.py\npython evaluate.py", language="bash")
    else:
        R = get_results(_results_stamp())
        grid = R["grid"]
        sel = R["selection"]
        hist = R["histories"]
        vp = R["val_predictions"]
        bval = R["baselines_val"].set_index("Modelo")
        tm = R["test_metrics"]
        SEL = sel["ID"]
        p_mae_val = bval.loc[ev.PERSIST, "MAE"]
        rnn_test = tm[tm["Modelo"].str.startswith("RNN")].iloc[0] if tm is not None else None

        # ------------------------------------------------------------ Recorrido en 4 pasos
        st.markdown("##### El recorrido de esta parte en 4 pasos")
        f1, f2, f3, f4 = st.columns(4)
        with f1.container(border=True):
            st.markdown("**① Cómo se valida**")
            st.metric("Partición cronológica", "70 / 15 / 15")
            st.caption("Sin mezclar el orden del tiempo: se entrena con el pasado y se evalúa con el futuro.")
        with f2.container(border=True):
            st.markdown("**② Qué se probó**")
            st.metric("RNN entrenadas", f"{len(grid)} configuraciones")
            st.caption(f"3 ventanas L × 3 tamaños H · {grid['Tiempo (s)'].sum() / 60:.1f} min en CPU.")
        with f3.container(border=True):
            st.markdown("**③ Cuál se eligió**")
            st.metric("Mejor en validación", f"{SEL} · L={sel['L']}, H={sel['H']}")
            st.caption(f"MAE = {sel['val_mae']:.1f} bicicletas/h, el menor de las {len(grid)}.")
        with f4.container(border=True):
            st.markdown("**④ Resultado final**")
            if rnn_test is not None:
                st.metric("MAE en prueba", f"{rnn_test['MAE']:.1f}",
                          f"{-rnn_test['Mejora MAE vs. persistencia (%)']:.0f} % vs. persistencia",
                          delta_color="inverse")
                st.caption("Evaluado una sola vez sobre 2 607 h nunca vistas.")
            else:
                st.metric("MAE en prueba", "pendiente")
                st.caption("Ejecuta `python evaluate.py`.")

        sub = st.tabs([
            "① Protocolo y reproducibilidad (4.7–4.8)",
            "② Experimentos (5.1)",
            "③ Curvas de pérdida (5.2)",
            "④ Selección (5.3)",
            "⑤ Prueba final (5.4)",
        ])

        # ---------------------------------------------------------------- 4.7 / 4.8
        with sub[0]:
            st.subheader("4.7 Protocolo de validación")
            _howto("cómo se reparten los 2 años de datos y para qué se usa cada tramo.",
                   "el tiempo avanza de izquierda a derecha. La red aprende con el bloque azul, se ajusta y "
                   "elige con el naranja, y el rojo se abre una sola vez al final.")
            _show(ev.fig_protocol())
            c_a, c_b, c_c = st.columns(3)
            with c_a.container(border=True):
                st.markdown("**🔵 Entrenamiento**")
                st.markdown(f"{I_TR:,} h · 2011-01 a 2012-05\n\nAjusta los pesos de la red y el escalador MinMax.")
            with c_b.container(border=True):
                st.markdown("**🟠 Validación**")
                st.markdown(f"{I_VA - I_TR:,} h · 2012-05 a 2012-09\n\nDecide cuándo parar (early stopping) "
                            "y qué configuración gana.")
            with c_c.container(border=True):
                st.markdown("**🔴 Prueba**")
                st.markdown(f"{len(df) - I_VA:,} h · 2012-09 a 2012-12\n\nSe usa **una sola vez** con el "
                            "modelo ya elegido.")

            st.markdown("#### ¿Cómo se evita mirar el futuro?")
            _howto("una muestra real de validación: la ventana que recibe la red y la hora que debe adivinar.",
                   "mueve el control para recorrer muestras. En las primeras, parte de la ventana cae en "
                   "entrenamiento: es pasado legítimo. La estrella (t+1) siempre queda a la derecha de la ventana.")
            cv1, cv2 = st.columns([1, 3])
            v_L = cv1.select_slider("Ventana L", WINDOW_SIZES, value=sel["L"], key="p4_vL")
            v_j = cv2.slider("Muestra j de validación (0 = primera hora de validación)", 0, 240, 6, key="p4_vj")
            _show(ev.fig_window_example(df, I_TR, v_L, v_j))
            tgt = I_TR + v_j
            st.caption(f"Ventana: {df.timestamp[tgt - v_L]:%Y-%m-%d %H:00} → {df.timestamp[tgt - 1]:%Y-%m-%d %H:00} "
                       f"(t) · Objetivo: {df.timestamp[tgt]:%Y-%m-%d %H:00} (t+1) · "
                       f"{max(0, v_L - v_j)} de {v_L} horas vienen del tramo de entrenamiento.")
            with st.expander("Ver las 4 reglas anti-fuga aplicadas"):
                st.markdown(
                    "1. **Escalador** ajustado solo con entrenamiento; validación y prueba usan esos mismos límites.\n"
                    "2. **Entrada ≤ t, objetivo = t+1:** ninguna variable de la hora a predecir entra en la ventana.\n"
                    "3. **Contexto solo pasado:** las primeras ventanas de validación/prueba toman las últimas L "
                    "horas del tramo anterior, nunca horas posteriores.\n"
                    "4. **Barajado solo dentro de entrenamiento:** la red reinicia su memoria ($h_0 = 0$) en cada "
                    "ventana, así que mezclar ventanas de entrenamiento no cruza información entre tramos."
                )
            _key("la validación es un bloque cronológico; la red nunca ve una hora posterior a la que "
                 "pronostica y la prueba queda intacta hasta el final.")

            st.divider()
            st.subheader("4.8 Reproducibilidad")
            env = R["environment"] or {}
            rp = R["reproducibility"]
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Semilla fija", f"SEED = {SEED}")
            r2.metric("Dispositivo", env.get("Dispositivo", "CPU"))
            r3.metric("PyTorch", env.get("PyTorch", "—"))
            if rp:
                r4.metric(f"Reentreno de {rp['ID']}", "✅ Idéntico" if rp["Idéntico"] else "⚠️ Difiere",
                          f"MAE {rp['MAE grid']:.2f} → {rp['MAE reentrenado']:.2f}", delta_color="off")
            else:
                r4.metric("Reentreno de control", "—")
            _key("con la misma semilla y versiones, volver a entrenar produce exactamente las mismas cifras.")
            with st.expander("Ver versiones, configuración y comandos para reproducir"):
                import importlib.metadata as _md

                def _ver(pkg):
                    try:
                        return _md.version(pkg)
                    except _md.PackageNotFoundError:
                        return "—"

                now = {"PyTorch": torch.__version__, "NumPy": np.__version__, "pandas": pd.__version__,
                       "Matplotlib": _ver("matplotlib"), "Streamlit": _ver("streamlit")}
                env_df = pd.DataFrame(
                    [(k, v, now.get(k, "")) for k, v in env.items() if k != "Comando"],
                    columns=["Elemento", "Valor en la ejecución de results/", "Versión en esta sesión"],
                )
                st.dataframe(env_df, hide_index=True, use_container_width=True)
                st.json(R["train_config"], expanded=False)
                st.code(
                    "pip install -r requirements.txt\n"
                    "python experiments.py   # entrena C1–C9, guarda results/ y verifica reproducibilidad\n"
                    "python evaluate.py      # evaluación única en prueba + figuras\n"
                    "streamlit run app.py",
                    language="bash",
                )

        # ---------------------------------------------------------------- 5.1
        with sub[1]:
            st.subheader("5.1 Experimentos realizados")
            _howto("el error de validación de las 9 RNN y de las 4 líneas base, ordenados de peor (arriba) a "
                   "mejor (abajo).",
                   "barras más cortas = menos error. El MAE dice cuántas bicicletas por hora se equivoca el "
                   "modelo en promedio. Verde = RNN (★ la elegida); otros colores = líneas base.")
            m_choice = st.radio("Métrica:", ["MAE", "RMSE"], horizontal=True, key="p4_rankmetric")
            _show(ev.fig_model_ranking(grid, bval[["MAE", "RMSE"]], SEL, m_choice))
            worst = grid.loc[grid["MAE val"].idxmax()]
            _key(f"todas las RNN superan a todas las líneas base. Incluso la peor ({worst['ID']}, MAE "
                 f"{worst['MAE val']:.1f}) se equivoca {(1 - worst['MAE val'] / p_mae_val) * 100:.0f} % menos que "
                 f"la persistencia y {(1 - worst['MAE val'] / bval.loc[ev.RIDGE, 'MAE']) * 100:.0f} % menos que Ridge.")

            st.markdown("#### ¿Qué combinación de L y H funciona mejor?")
            _howto("cómo cambia el error al variar la historia (L) y el tamaño de la memoria (H).",
                   "en el mapa, el color más claro es el mejor. En las líneas, cada color es un H; el punto "
                   "más bajo indica la mejor ventana.")
            cg1, cg2 = st.columns(2)
            with cg1:
                _show(ev.fig_grid_heatmap(grid, "MAE val", SEL))
            with cg2:
                _show(ev.fig_effects(grid))
            mL = grid.groupby("Ventana L")["MAE val"].mean()
            _key(f"la ventana de 24 h es la mejor (MAE medio {mL[24]:.1f} frente a {mL[12]:.1f} con 12 h y "
                 f"{mL[48]:.1f} con 48 h): un día completo contiene el ciclo diario y 48 pasos ya cuestan a la "
                 "RNN simple. Más neuronas (H) ayudan un poco.")

            with st.expander("Ver tablas: plan de experimentos A–D y rejilla C1–C9 completa"):
                d_rows = grid[grid["ID"].isin(["C5", "C6", "C8", "C9"])].sort_values("MAE val")
                best_d = d_rows.iloc[0]
                c1_row = grid[grid["ID"] == "C1"].iloc[0]
                exp = pd.DataFrame([
                    ("A", "Persistencia: último cnt", "Línea base",
                     *bval.loc[ev.PERSIST, ["MAE", "RMSE", "MASE"]]),
                    ("B", "Promedio misma hora, últimos 7 días", "Línea base estacional",
                     *bval.loc[ev.SEASONAL, ["MAE", "RMSE", "MASE"]]),
                    ("C", "RNN L = 12, H = 16 (C1)", "Memoria corta",
                     c1_row["MAE val"], c1_row["RMSE val"], c1_row["MASE val"]),
                    ("D", f"RNN L = 24/48, H = 32/64 (mejor: {best_d['ID']})", "Dependencias más largas",
                     best_d["MAE val"], best_d["RMSE val"], best_d["MASE val"]),
                    ("Ref.", "Ridge sobre rezagos", "Referencia lineal exigente",
                     *bval.loc[ev.RIDGE, ["MAE", "RMSE", "MASE"]]),
                ], columns=["Exp.", "Configuración", "Finalidad", "MAE", "RMSE", "MASE"])
                st.dataframe(exp.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "MASE": "{:.3f}"}),
                             hide_index=True, use_container_width=True)

                def _hl(row):
                    return ["background-color: #E3F1E8; font-weight: 600" if row["ID"] == SEL else "" for _ in row]

                show_cols = ["ID", "Ventana L", "Hidden H", "Parámetros", "Tiempo (s)", "Épocas", "Mejor época",
                             "MAE val", "RMSE val", "MASE val"]
                st.dataframe(
                    grid[show_cols].style.apply(_hl, axis=1).format(
                        {"MAE val": "{:.2f}", "RMSE val": "{:.2f}", "MASE val": "{:.3f}", "Tiempo (s)": "{:.1f}",
                         "Parámetros": "{:,}"}),
                    hide_index=True, use_container_width=True,
                )
                st.caption("MASE < 1 = mejor que la persistencia de entrenamiento (MAE ingenuo = 55.60).")

        # ---------------------------------------------------------------- 5.2
        with sub[2]:
            st.subheader("5.2 Curvas de pérdida y seguimiento")
            _howto("cómo bajó el error de la red época a época, en los datos que ve (azul) y en datos nuevos "
                   "(naranja).",
                   "si ambas curvas bajan, la red aprende. Cuando la naranja deja de bajar, se detiene el "
                   "entrenamiento (early stopping) y se guardan los pesos de la mejor época (línea verde).")
            l1, l2 = st.columns([1, 3])
            f_cfg = l1.selectbox("Configuración", grid["ID"].tolist(), index=grid["ID"].tolist().index(SEL),
                                 key="p4_lossid")
            r_cfg = grid[grid["ID"] == f_cfg].iloc[0]
            with l2:
                k1, k2, k3 = st.columns(3)
                k1.metric("Épocas ejecutadas", int(r_cfg["Épocas"]))
                k2.metric("Mejor época", int(r_cfg["Mejor época"]))
                be = int(r_cfg["Mejor época"]) - 1
                k3.metric("Brecha val/train", f"{hist[f_cfg]['val_loss'][be] / hist[f_cfg]['train_loss'][be]:.1f}×")
            _show(ev.fig_loss_focus(hist[f_cfg], f_cfg, int(r_cfg["Mejor época"])))
            _key("las curvas convergen sin explotar (el clipping funciona). La validación queda ≈ 2× sobre "
                 "entrenamiento: parte es sobreajuste leve y parte el verano de 2012, con más demanda que lo "
                 "visto al entrenar. Con L = 48 el early stopping llega antes y el error es mayor.")
            with st.expander("Comparar las 9 configuraciones"):
                _show(ev.fig_loss_grid(hist, grid, SEL))

            st.markdown("#### ¿Cómo se ve el pronóstico en validación?")
            _howto("una semana de validación: demanda real (negro) frente a la RNN y dos líneas base.",
                   "la RNN (verde) se pega a la curva real. La persistencia (naranja) llega siempre una hora "
                   "tarde y el promedio estacional (azul) falla cuando cambia el tipo de día.")
            cf1, cf2, cf3 = st.columns([1, 1, 3])
            f_id = cf1.selectbox("Configuración", grid["ID"].tolist(), index=grid["ID"].tolist().index(SEL),
                                 key="p4_fid")
            n_days = cf2.radio("Días", [3, 7], index=1, horizontal=True, key="p4_days")
            f_s = cf3.slider("Inicio (hora relativa en validación)", 0, len(vp) - 168,
                             ev.best_week_start(vp["timestamp"]), step=24, key="p4_fs")
            w = slice(f_s, f_s + 24 * n_days)
            _show(ev.fig_forecast(
                vp["timestamp"].iloc[w], vp["y_true"].iloc[w].to_numpy(),
                {f"RNN {f_id}": vp[f_id].iloc[w].to_numpy(),
                 "A. Persistencia": vp["persistencia"].iloc[w].to_numpy(),
                 "B. Promedio estacional (k=7)": vp["promedio_estacional"].iloc[w].to_numpy()},
                f"Validación · {n_days} días desde {vp['timestamp'].iloc[f_s]:%Y-%m-%d}"))
            yw = vp["y_true"].iloc[w].to_numpy()
            k1, k2, k3 = st.columns(3)
            k1.metric(f"MAE del tramo · RNN {f_id}", f"{mae(yw, vp[f_id].iloc[w]):.1f}")
            k2.metric("MAE del tramo · Persistencia", f"{mae(yw, vp['persistencia'].iloc[w]):.1f}")
            k3.metric("MAE del tramo · Prom. estacional", f"{mae(yw, vp['promedio_estacional'].iloc[w]):.1f}")

        # ---------------------------------------------------------------- 5.3
        with sub[3]:
            st.subheader("5.3 Selección de configuración")
            st.info(f"**📏 Regla de selección (fijada antes de ver resultados):** {sel['criterio']}. "
                    "Solo se usa validación; la prueba no interviene.")
            rk = grid.sort_values(["MAE val", "RMSE val", "Parámetros"]).reset_index(drop=True)
            podium = st.columns(3)
            for col, (i, r), medal in zip(podium, rk.head(3).iterrows(), ["🥇", "🥈", "🥉"]):
                with col.container(border=True):
                    st.markdown(f"### {medal} {r['ID']} · L={r['Ventana L']}, H={r['Hidden H']}")
                    st.metric("MAE validación", f"{r['MAE val']:.2f}",
                              None if i == 0 else f"+{r['MAE val'] - sel['val_mae']:.2f} vs. {SEL}",
                              delta_color="inverse")
                    st.caption(f"RMSE {r['RMSE val']:.2f} · {int(r['Parámetros']):,} parámetros · "
                               f"{r['Tiempo (s)']:.0f} s")

            cs1, cs2 = st.columns(2)
            with cs1:
                _howto("error frente a tamaño del modelo; el tamaño del punto es el tiempo de entrenamiento.",
                       "lo ideal está abajo a la izquierda (poco error, pocos parámetros).")
                _show(ev.fig_complexity(grid, SEL))
            with cs2:
                _howto(f"cuánto reduce {SEL} el error de cada línea base en validación.",
                       "barra más larga = mayor ventaja de la RNN.")
                _show(ev.fig_gain_bars(sel["val_mae"], {k: bval.loc[k, "MAE"] for k in
                                                         [ev.PERSIST, ev.SEASONAL, ev.SEASONAL_1, ev.RIDGE]},
                                       f"Ventaja de {SEL} en validación"))
            second = rk.iloc[1]
            _key(f"{SEL} tiene el menor MAE y también el menor RMSE, por eso se elige. {second['ID']} queda a "
                 f"solo {second['MAE val'] - sel['val_mae']:.2f} bicicletas/h con menos parámetros: la ventaja es "
                 "pequeña y con una sola semilla no puede asegurarse que sea significativa.")
            with st.expander("Ver ranking completo"):
                rk2 = rk.copy()
                rk2.index = rk2.index + 1
                rk2["Δ MAE vs. elegida"] = rk2["MAE val"] - sel["val_mae"]
                st.dataframe(rk2[["ID", "Ventana L", "Hidden H", "Parámetros", "Tiempo (s)", "MAE val",
                                  "RMSE val", "Δ MAE vs. elegida"]].style.format(
                    {"MAE val": "{:.2f}", "RMSE val": "{:.2f}", "Δ MAE vs. elegida": "{:+.2f}",
                     "Tiempo (s)": "{:.1f}"}), use_container_width=True)

        # ---------------------------------------------------------------- 5.4
        with sub[4]:
            st.subheader("5.4 Evaluación final en el conjunto de prueba")
            if not status["test"]:
                st.warning("Falta la evaluación en prueba. Ejecuta `python evaluate.py`.")
            else:
                tp = R["test_predictions"]
                b_row = tm[tm["Modelo"] == ev.SEASONAL].iloc[0]
                r_row = tm[tm["Modelo"] == ev.RIDGE].iloc[0]
                t1, t2, t3, t4 = st.columns(4)
                t1.metric("MAE RNN en prueba", f"{rnn_test['MAE']:.1f}", f"val: {sel['val_mae']:.1f}",
                          delta_color="off")
                t2.metric("vs. Persistencia", f"−{rnn_test['Mejora MAE vs. persistencia (%)']:.0f} % de error")
                t3.metric("vs. Promedio estacional", f"−{(1 - rnn_test['MAE'] / b_row['MAE']) * 100:.0f} % de error")
                t4.metric("vs. Ridge", f"−{(1 - rnn_test['MAE'] / r_row['MAE']) * 100:.0f} % de error")

                _howto("el error de cada modelo sobre 2 607 horas (sep–dic 2012) que nadie usó antes.",
                       "barra más corta = mejor. Entre paréntesis, cuánto mejora o empeora cada modelo frente a "
                       "la persistencia.")
                _show(ev.fig_test_ranking(tm))
                _key(f"la RNN mantiene en prueba el nivel de validación (MAE {sel['val_mae']:.1f} → "
                     f"{rnn_test['MAE']:.1f}) y reduce el error de la persistencia en "
                     f"{rnn_test['Mejora MAE vs. persistencia (%)']:.0f} %. No hay señal de que la selección haya "
                     "sobreajustado la validación.")

                st.markdown("#### Pronóstico vs. realidad")
                _howto("cada punto es una hora de prueba: en el eje x la demanda real y en el y la pronosticada.",
                       "si el pronóstico fuera perfecto, todos los puntos caerían en la diagonal. La nube de la RNN "
                       "es mucho más estrecha que la de la persistencia.")
                _show(ev.fig_real_vs_pred(tp))

                _howto("una semana de prueba con la demanda real y los pronósticos.",
                       "elige qué líneas base superponer y mueve la semana. Observa en qué horas falla cada modelo.")
                ct1, ct2, ct3 = st.columns([2, 1, 3])
                opts = {"A. Persistencia": "persistencia", "B. Promedio estacional (k=7)": "promedio_estacional",
                        "B0. Día anterior (k=1)": "misma_hora_dia_anterior", "C. Ridge rezagos": "ridge"}
                chosen = ct1.multiselect("Líneas base a superponer:", list(opts),
                                         default=["A. Persistencia"], key="p4_tb")
                t_days = ct2.radio("Días", [3, 7], index=1, horizontal=True, key="p4_tdays")
                t_s = ct3.slider("Inicio (hora relativa en prueba)", 0, len(tp) - 168,
                                 ev.best_week_start(tp["timestamp"]), step=24, key="p4_ts")
                w = slice(t_s, t_s + 24 * t_days)
                preds = {f"RNN {SEL}": tp["rnn"].iloc[w].to_numpy()}
                preds.update({k: tp[opts[k]].iloc[w].to_numpy() for k in chosen})
                _show(ev.fig_forecast(tp["timestamp"].iloc[w], tp["y_true"].iloc[w].to_numpy(), preds,
                                      f"Prueba · {t_days} días desde {tp['timestamp'].iloc[t_s]:%Y-%m-%d}"))
                st.caption("Tip: alrededor del 29-oct-2012 (huracán Sandy) la demanda cae casi a cero; "
                           "el promedio estacional arrastra ese día atípico durante una semana.")

                with st.expander("Ver tabla completa de prueba y comparación validación → prueba"):
                    st.dataframe(tm.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "MASE": "{:.3f}",
                                                  "Mejora MAE vs. persistencia (%)": "{:+.2f} %",
                                                  "Mejora RMSE vs. persistencia (%)": "{:+.2f} %"}),
                                 hide_index=True, use_container_width=True)
                    gap = pd.DataFrame({
                        "Métrica": ["MAE", "RMSE", "MASE"],
                        "Validación": [sel["val_mae"], sel["val_rmse"], sel["val_mase"]],
                        "Prueba": [rnn_test["MAE"], rnn_test["RMSE"], rnn_test["MASE"]],
                    })
                    gap["Cambio (%)"] = (gap["Prueba"] / gap["Validación"] - 1) * 100
                    st.dataframe(gap.style.format({"Validación": "{:.2f}", "Prueba": "{:.2f}",
                                                   "Cambio (%)": "{:+.1f} %"}),
                                 hide_index=True, use_container_width=True)
                    st.caption("Las predicciones completas quedan en `results/test_predictions.csv` para el "
                               "análisis de error por hora de la Parte 5.")
