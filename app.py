"""Streamlit · Pronóstico horario de alquiler de bicicletas con una RNN simple (Grupo 03)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from config import DIAGRAM_PATH, FEATURES, SEED, TARGET, TRAIN_FRAC, VAL_FRAC, set_seeds
from data import VARIABLES, load_hour
from features import EXCLUDED_LEAKAGE, EXCLUDED_REDUNDANT, add_cyclic_features
from metrics import (mae, mae_naive_train, mase, persistence_forecast,
                     rmse, rnn_param_count, same_hour_mean_forecast)
from preprocessing import (TemporalMinMaxScaler, create_sliding_windows,
                           split_dataframe, temporal_split_indices)

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

# Solo dos pestañas principales limpias
tabs = st.tabs([
    "📘 PARTE 1: Problema, Fundamentos y Datos (Del 1 al 3.2)",
    "📊 PARTE 2: Entradas, EDA y Preprocesamiento (Del 3.3 al 4.2)"
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
