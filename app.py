"""Tablero en Streamlit para pronóstico horario del Precio de Bolsa de Energía (PBE).

La aplicación descarga datos históricos desde SIMEM (dataset EC6945), los preprocesa a frecuencia horaria
y entrena dos enfoques de predicción:

1) HMM + ML: un Modelo Oculto de Markov (HMM) gaussiano sobre retornos logarítmicos para inferir regímenes,
   y una mezcla de expertos con modelos de boosting (uno global y opcionalmente uno por régimen).
2) MC-horaria: Modelo de Cadenas de Markov No Homogéneas por Hora sobre estados discretos del precio.

La app realiza:
- Visualización de la serie y del HMM (matriz de transición, regímenes y probabilidades).
- Prueba en las últimas horas para comparar métricas.
- Pronostica para las próximas 24 horas e intenta superponer el precio real (indepeindientemente si ya existe).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Modelo HMM (hmmlearn)
from hmmlearn.hmm import GaussianHMM

# Cliente de datos SIMEM
from pydataxm.pydatasimem import ReadSIMEM

# Regressor para la mezcla de expertos
from sklearn.ensemble import HistGradientBoostingRegressor


# Configuración básica de la página Streamlit
st.set_page_config(
    page_title="Predicción PBE horario (HMM vs MC-horaria)",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Predicción robusta del Precio de Bolsa horario (HMM vs MC-horaria)")
st.caption(
    "Datos: SIMEM (EC6945, PB_Nal/PB_Int/PB_Tie).\n"
    "Modelo 1: Modelo Oculto de Markov (HMM) + mezcla de expertos (boosting por régimen).\n"
    "Modelo 2: Modelo de Cadenas de Markov No Homogéneas por Hora (MC-horaria) sobre estados de precio."
)


# Identificador del dataset en SIMEM
DATASET_ID = "EC6945"

# Rango por defecto para facilitar la ejecución sin configuración adicional
DEFAULT_START = "2024-02-04"
DEFAULT_END = "2024-03-04"

# Horizonte por defecto para el pronóstico futuro
DEFAULT_HORIZON = 24


@st.cache_data(show_spinner=False)
def _infer_datetime_col(df: pd.DataFrame) -> Optional[str]:
    """Infiera la columna que representa fecha/hora.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame crudo (por ejemplo, retornado por SIMEM).

    Retorna
    -------
    Optional[str]
        Nombre de la columna que se puede convertir a datetime, o None si no se identifica.
    """
    # Primero se prueban nombres típicos
    preferred = ["FechaHora", "fechaHora", "fecha_hora", "Fecha", "fecha"]
    for c in preferred:
        if c in df.columns:
            try:
                pd.to_datetime(df[c])
                return c
            except Exception:
                pass

    # Si no hay coincidencias obvias, se intenta con todas las columnas
    for col in df.columns:
        try:
            pd.to_datetime(df[col])
            return col
        except Exception:
            continue

    return None


@st.cache_data(show_spinner=False)
def _infer_numeric_col(df: pd.DataFrame) -> Optional[str]:
    """Infiera la columna numérica que contiene el precio.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame crudo.

    Retorna
    -------
    Optional[str]
        Nombre de una columna convertible a numérico, o None si no se encuentra.
    """
    # Se priorizan nombres comunes para el precio
    preferred = [
        "PrecioBolsa",
        "precioBolsa",
        "precio_bolsa",
        "Precio",
        "precio",
        "PBE",
        "pbe",
        "Valor",
    ]
    for c in preferred:
        if c in df.columns:
            try:
                pd.to_numeric(df[c])
                return c
            except Exception:
                pass

    # Fallback: primera columna numérica ya tipada como número
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    return num[0] if num else None


@st.cache_data(show_spinner=False)
def preprocess_price(
    df: pd.DataFrame,
    dt_col: str,
    val_col: str,
    tz: str = "America/Bogota",
) -> pd.DataFrame:
    """Convierte datos crudos a una serie horaria limpia de precios.

    Pasos clave:
    - Convierte la columna temporal a datetime.
    - Normaliza/convierte a la zona horaria objetivo.
    - Convierte el valor a numérico y elimina nulos.
    - Re-muestrea a frecuencia horaria (promedio) y rellena huecos con forward-fill.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con columnas de fecha y valor.
    dt_col : str
        Nombre de la columna temporal.
    val_col : str
        Nombre de la columna numérica (precio).
    tz : str
        Zona horaria destino para el índice.

    Retorna
    -------
    pd.DataFrame
        DataFrame con índice datetime y una columna llamada 'precio' a frecuencia horaria.
    """
    df = df.copy()

    # Interpreta la fecha/hora; se asume UTC si viene con información compatible
    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce", utc=True)
    df = df.dropna(subset=[dt_col])

    # Intenta convertir de UTC a la zona horaria objetivo; si falla, se localiza
    try:
        df[dt_col] = df[dt_col].dt.tz_convert(tz)
    except Exception:
        df[dt_col] = df[dt_col].dt.tz_localize(
            tz,
            nonexistent="shift_forward",
            ambiguous="NaT",
        )

    # Se asegura que el precio sea numérico y elimina registros inválidos
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    df = df.dropna(subset=[val_col])

    # Indexar por fecha, ordenar y dejar solo la columna de precio
    df = df.set_index(dt_col).sort_index()[[val_col]]

    # Re-muestreo a frecuencia horaria para uniformizar el modelo
    df = df.resample("H").mean().ffill()

    df.columns = ["precio"]
    df = df.dropna()
    return df


def mae(y_true, y_pred) -> float:
    """Calcula el error absoluto medio (MAE)."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    """Calcula la raíz del error cuadrático medio (RMSE)."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred) -> float:
    """Calcula el MAPE (%), excluyendo verdaderos iguales a cero para evitar división por cero."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def smape(y_true, y_pred) -> float:
    """Calcula el sMAPE (%), que normaliza por |y|+|ŷ| para mayor estabilidad."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom != 0
    return float(np.mean(2.0 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]) * 100)


def _chunk_dates(start: datetime, end: datetime, max_days: int = 31) -> List[Tuple[str, str]]:
    """Divide un rango de fechas en segmentos para consultas paginadas a SIMEM.

    Parámetros
    ----------
    start : datetime
        Fecha inicial inclusiva.
    end : datetime
        Fecha final inclusiva.
    max_days : int
        Tamaño máximo del segmento (en días) para evitar respuestas muy grandes.

    Retorna
    -------
    List[Tuple[str, str]]
        Lista de (fecha_inicio, fecha_fin) en formato YYYY-MM-DD.
    """
    chunks: List[Tuple[str, str]] = []
    cur = start
    while cur <= end:
        nxt = min(cur + timedelta(days=max_days - 1), end)
        chunks.append((cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")))
        cur = nxt + timedelta(days=1)
    return chunks


@st.cache_data(show_spinner=True)
def simem_fetch_ec6945(start_date: str, end_date: str) -> pd.DataFrame:
    """Descarga EC6945 desde SIMEM en el rango solicitado.

    La descarga se hace por "chunks" para ser más robusta ante límites de tamaño y latencia.

    Parámetros
    ----------
    start_date : str
        Fecha inicio (YYYY-MM-DD).
    end_date : str
        Fecha fin (YYYY-MM-DD).

    Retorna
    -------
    pd.DataFrame
        Datos concatenados; DataFrame vacío si no hubo resultados.
    """
    sdt = datetime.fromisoformat(start_date)
    edt = datetime.fromisoformat(end_date)

    dfs: List[pd.DataFrame] = []
    for s, e in _chunk_dates(sdt, edt, 31):
        # ReadSIMEM maneja la consulta al dataset por rango de fechas
        reader = ReadSIMEM(DATASET_ID, s, e)
        df = reader.main(filter=False)
        if isinstance(df, pd.DataFrame) and not df.empty:
            dfs.append(df)

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def make_feature_table(prices: pd.Series) -> pd.DataFrame:
    """Construye una tabla de características para el modelo de regresión.

    Las características se basan solo en la serie de precios y en la estacionalidad temporal:
    - Hora y día de la semana.
    - Componentes seno/coseno para capturar periodicidad.
    - Rezagos (1h, 2h, 24h).
    - Promedios móviles (24h y 168h) usando información pasada.

    Parámetros
    ----------
    prices : pd.Series
        Serie temporal de precio con índice datetime.

    Retorna
    -------
    pd.DataFrame
        Tabla con columna 'precio' y columnas de características, sin nulos.
    """
    df = prices.to_frame("precio").copy()

    # Variables calendario
    df["hour"] = df.index.hour
    df["dow"] = df.index.dayofweek

    # Codificación cíclica para evitar discontinuidades (ej: hora 23 vs 0)
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["sin_dow"] = np.sin(2 * np.pi * df["dow"] / 7.0)
    df["cos_dow"] = np.cos(2 * np.pi * df["dow"] / 7.0)

    # Rezagos de corto y mediano plazo
    for lag in [1, 2, 24]:
        df[f"lag{lag}"] = df["precio"].shift(lag)

    # Promedios móviles usando el pasado (shift(1) evita fuga de información)
    df["roll24"] = df["precio"].shift(1).rolling(24).mean()
    df["roll168"] = df["precio"].shift(1).rolling(168).mean()

    # Se eliminan filas con nulos creados por rezagos y ventanas
    df = df.dropna()
    return df


def build_future_feature_row(price_series_ext: pd.Series, t: pd.Timestamp) -> pd.Series:
    """Construye las características para un instante futuro t.

    Se usa una serie extendida (histórico + predicciones previas) para poder
    calcular rezagos y promedios móviles durante el pronóstico multi-paso.

    Parámetros
    ----------
    price_series_ext : pd.Series
        Serie extendida con índice datetime y valores de precio.
    t : pd.Timestamp
        Instante futuro para el cual se requiere el vector de características.

    Retorna
    -------
    pd.Series
        Serie con las mismas columnas de características que produce make_feature_table.
    """
    row: dict = {}

    # Variables calendario para el instante futuro
    hour = t.hour
    dow = t.dayofweek
    row["hour"] = hour
    row["dow"] = dow
    row["sin_hour"] = np.sin(2 * np.pi * hour / 24.0)
    row["cos_hour"] = np.cos(2 * np.pi * hour / 24.0)
    row["sin_dow"] = np.sin(2 * np.pi * dow / 7.0)
    row["cos_dow"] = np.cos(2 * np.pi * dow / 7.0)

    # Rezagos: se consultan precios pasados (reales o predichos)
    for lag in [1, 2, 24]:
        tlag = t - pd.Timedelta(hours=lag)
        if tlag in price_series_ext.index:
            row[f"lag{lag}"] = float(price_series_ext.loc[tlag])
        else:
            # Si por alguna razón el índice exacto no está, se usa el último disponible
            row[f"lag{lag}"] = float(price_series_ext.iloc[-1])

    # Ventanas móviles: se calculan sobre el pasado inmediato
    past = price_series_ext.loc[: t - pd.Timedelta(hours=1)]
    if len(past) > 0:
        row["roll24"] = float(past.iloc[-24:].mean()) if len(past) >= 24 else float(past.mean())
        row["roll168"] = float(past.iloc[-168:].mean()) if len(past) >= 168 else row["roll24"]
    else:
        # Caso extremo: sin pasado, usar la media como valor de respaldo
        row["roll24"] = float(price_series_ext.mean())
        row["roll168"] = row["roll24"]

    return pd.Series(row)


def fit_hmm_and_features(
    price_series: pd.Series,
    n_states: int,
) -> Tuple[GaussianHMM, pd.DataFrame, pd.DataFrame]:
    """Ajusta un HMM gaussiano sobre retornos y construya tablas auxiliares.

    El HMM no se ajusta directamente sobre el precio (no estacionario), sino sobre retornos
    logarítmicos para reducir efectos de escala y mejorar estabilidad.

    Retorna:
    - hmm: modelo ajustado
    - df_feat: tabla de características con columna 'state' (estado Viterbi)
    - gamma_df: probabilidades a posteriori por estado (predict_proba)
    """
    # Construir características supervisadas (para el regresor)
    df_feat = make_feature_table(price_series)

    # Retornos logarítmicos alineados con el índice de características
    logp = np.log(price_series)
    ret = logp.diff()
    ret = ret.reindex(df_feat.index)
    ret = ret.dropna()
    df_feat = df_feat.loc[ret.index]

    # hmmlearn espera un arreglo 2D: (n_muestras, n_variables)
    obs = ret.values.reshape(-1, 1)

    # HMM gaussiano con covarianza diagonal para el caso univariado
    hmm = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=200,
        random_state=42,
    )
    hmm.fit(obs)

    # gamma: P(S_t = k | observaciones) para cada t
    gamma = hmm.predict_proba(obs)
    # states: trayectoria más probable (Viterbi)
    states = hmm.predict(obs)

    gamma_df = pd.DataFrame(
        gamma,
        index=df_feat.index,
        columns=[f"p_state_{k}" for k in range(n_states)],
    )

    df_feat = df_feat.copy()
    df_feat["state"] = states

    return hmm, df_feat, gamma_df


def train_state_regressors(
    df_feat: pd.DataFrame,
    n_states: int,
    horizon: int,
):
    """Entrena un modelo global y modelos por régimen para la mezcla de expertos.

    Estrategia:
    - Se reserva un bloque final de prueba para la retroprueba.
    - Se entrena un modelo global con todos los datos de entrenamiento.
    - Para cada régimen, si hay suficientes muestras, se entrena un modelo específico.
      Si no, ese régimen cae al modelo global al predecir.

    Parámetros
    ----------
    df_feat : pd.DataFrame
        Tabla con 'precio', características y 'state'.
    n_states : int
        Número de regímenes del HMM.
    horizon : int
        Horizonte sugerido (en horas) para el bloque de retroprueba.

    Retorna
    -------
    global_model, models, feature_cols, test_info
        - global_model: HistGradientBoostingRegressor entrenado.
        - models: dict con modelo por estado o None.
        - feature_cols: lista de columnas usadas como X (características).
        - test_info: tupla (X_test, y_test, test_index, test_len).
    """
    # Columnas predictoras; se excluye objetivo y etiqueta de estado
    feature_cols = [c for c in df_feat.columns if c not in ["precio", "state"]]

    X = df_feat[feature_cols]
    y = df_feat["precio"]

    # Tamaño del bloque de prueba: se limita para evitar dejar demasiado poco entrenamiento
    test_len = min(horizon, max(1, len(df_feat) // 5))
    X_train, X_test = X.iloc[:-test_len], X.iloc[-test_len:]
    y_train, y_test = y.iloc[:-test_len], y.iloc[-test_len:]
    states_train = df_feat["state"].iloc[:-test_len]
    test_index = X_test.index

    # Modelo global para todos los regímenes
    global_model = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.05,
        max_iter=400,
        random_state=42,
    )
    global_model.fit(X_train, y_train)

    # Modelos por estado (expertos)
    models = {}
    for k in range(n_states):
        mask_k = states_train == k
        # Umbral mínimo para evitar sobreajuste con pocos datos
        if mask_k.sum() < 50:
            models[k] = None
        else:
            mk = HistGradientBoostingRegressor(
                max_depth=5,
                learning_rate=0.05,
                max_iter=300,
                random_state=42,
            )
            mk.fit(X_train[mask_k], y_train[mask_k])
            models[k] = mk

    return global_model, models, feature_cols, (X_test, y_test, test_index, test_len)


def mixture_predict_row(
    x_row: pd.Series,
    probs: np.ndarray,
    models: dict,
    global_model: HistGradientBoostingRegressor,
    feature_cols: List[str],
) -> float:
    """Predice con mezcla de expertos ponderada por probabilidades del HMM.

    Fórmula:
        ŷ_t = Σ_k p_k(t) · f_k(x_t)

    Si un experto por régimen no existe (models[k] es None), se usa el modelo global.

    Parámetros
    ----------
    x_row : pd.Series
        Fila con características.
    probs : np.ndarray
        Vector de probabilidades por régimen (suma 1).
    models : dict
        Modelos por régimen.
    global_model : HistGradientBoostingRegressor
        Modelo de respaldo.
    feature_cols : List[str]
        Columnas a usar como entrada del modelo.

    Retorna
    -------
    float
        Predicción final ponderada.
    """
    preds: List[float] = []
    for k in range(len(probs)):
        model_k = models.get(k)
        if model_k is None:
            model_k = global_model

        # .predict espera un array 2D
        y_k = float(model_k.predict(x_row[feature_cols].values.reshape(1, -1))[0])
        preds.append(y_k)

    preds_arr = np.array(preds)
    return float(np.dot(probs, preds_arr))


def discretize_states(
    series: pd.Series,
    n_states: int,
    method: str = "quantiles",
):
    """Discretiza una serie continua en estados enteros.

    Métodos:
    - 'quantiles': cortes por cuantiles; robusto ante outliers.
    - 'kmeans': clustering 1D; los estados se reordenan por su media. (No usada)

    Parámetros
    ----------
    series : pd.Series
        Serie de precios.
    n_states : int
        Número de estados a generar.
    method : str
        'quantiles' o 'kmeans'.

    Retorna
    -------
    (states, info)
        - states: pd.Series con valores 0..n_states-1.
        - info: bins (quantiles) o centros (kmeans).
    """
    x = series.values.astype(float)

    if method == "quantiles":
        # Límites por cuantiles; np.unique evita cortes repetidos si hay valores constantes
        qs = np.linspace(0, 1, n_states + 1)
        bins = np.unique(np.quantile(x, qs))

        # Si hay muchos valores repetidos, el número efectivo de bins puede reducirse
        if len(bins) - 1 < n_states:
            n_states_eff = len(bins) - 1
            st.warning(
                f"Advertencia: solo se pudieron formar {n_states_eff} estados efectivos "
                "por cuantiles. Se usará ese número de estados."
            )
            n_states = n_states_eff

        # Extremos abiertos para garantizar inclusión total
        bins[0] = -np.inf
        bins[-1] = np.inf

        states = pd.cut(series, bins=bins, labels=False, include_lowest=True)
        states = states.astype(int)
        return states, bins

    if method == "kmeans":
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=n_states, n_init="auto", random_state=42)
        labels = km.fit_predict(x.reshape(-1, 1))

        # Se reordenan etiquetas según la media del precio en cada cluster
        order = np.argsort([x[labels == k].mean() for k in range(n_states)])
        remap = {old: new for new, old in enumerate(order)}
        states = pd.Series(labels, index=series.index).map(remap).astype(int)
        centers = km.cluster_centers_.ravel()[order]
        return states, centers

    raise ValueError("method debe ser 'quantiles' o 'kmeans'.")


def mc_hourly_fit(
    states: pd.Series,
    prices: pd.Series,
    n_states: int,
    n_hours: int = 24,
):
    """Ajusta la MC-horaria: matrices P^{(h)} por hora del día.

    Definición:
        P^{(h)}[i, j] = P(S_{t+1}=j | S_t=i, hora(t)=h)

    Detalles importantes:
    - Se usa suavizado de Laplace (inicialización en 1) para evitar filas con probabilidad cero.
    - Además, se calcula la media global del precio por estado para mapear probabilidades a valores.

    Parámetros
    ----------
    states : pd.Series
        Estados discretos (0..n_states-1) indexados por tiempo.
    prices : pd.Series
        Serie de precios alineada con states.
    n_states : int
        Número de estados.
    n_hours : int
        Número de horas del día (24).

    Retorna
    -------
    (P, state_means)
        - P: np.ndarray de forma (24, n_states, n_states).
        - state_means: np.ndarray con media de precio por estado.
    """
    states = states.astype(int)
    prices = prices.astype(float)

    # Conteos con suavizado Laplace para evitar probabilidades nulas
    counts = np.ones((n_hours, n_states, n_states), dtype=float)

    s_vals = states.values
    times = states.index

    # Acumular transiciones S_t -> S_{t+1} condicionadas a la hora de t
    for idx in range(len(states) - 1):
        i = s_vals[idx]
        j = s_vals[idx + 1]
        h = times[idx].hour
        if 0 <= i < n_states and 0 <= j < n_states:
            counts[h, i, j] += 1.0

    # Normalizar por filas para obtener probabilidades
    P = counts / counts.sum(axis=2, keepdims=True)

    # Media global de precio por estado; sirve para convertir estados a valores esperados
    state_means = np.zeros(n_states, dtype=float)
    for k in range(n_states):
        mask = states == k
        state_means[k] = prices[mask].mean() if mask.any() else prices.mean()

    return P, state_means


st.sidebar.header("⚙️ Parámetros de datos (SIMEM • EC6945)")

# Entradas de fecha; se convierten a str para la consulta
c1, c2 = st.sidebar.columns(2)
with c1:
    start_date = st.sidebar.date_input(
        "Fecha inicio",
        value=datetime.fromisoformat(DEFAULT_START),
    ).strftime("%Y-%m-%d")
with c2:
    end_date = st.sidebar.date_input(
        "Fecha fin",
        value=datetime.fromisoformat(DEFAULT_END),
    ).strftime("%Y-%m-%d")

# Selector de variable de precio
codigo_variable = st.sidebar.selectbox(
    "CodigoVariable",
    ["PB_Nal", "PB_Int", "PB_Tie"],
    index=0,
)

st.sidebar.header("🧠 Modelos")

# Parámetros del HMM y de la MC-horaria
n_states = st.sidebar.slider("Regímenes HMM", 2, 5, 3)
mc_n_states = st.sidebar.slider("Estados MC-horaria", 3, 10, 5)
mc_disc_method = st.sidebar.selectbox("Discretización MC-horaria", ["quantiles", "kmeans"], index=0)
horizon = st.sidebar.slider("Horizonte de retroprueba (horas)", 12, 48, 24)


if st.sidebar.button("🔄 Traer datos de SIMEM y entrenar modelos"):
    # Descarga datos crudos
    with st.spinner("Descargando EC6945 desde SIMEM..."):
        df_raw = simem_fetch_ec6945(start_date, end_date)

    if df_raw.empty:
        st.error("La consulta no devolvió registros. Revisa el rango de fechas.")
        st.stop()

    # Filtra por variable si existe la columna correspondiente
    if "CodigoVariable" in df_raw.columns:
        df_raw = df_raw[df_raw["CodigoVariable"] == codigo_variable].copy()
        if df_raw.empty:
            st.error(f"No hay registros para CodigoVariable = {codigo_variable}.")
            st.stop()
    else:
        st.warning("No se encontró la columna 'CodigoVariable'. Se usa todo el conjunto retornado.")

    # Infiere columnas de fecha y valor si no son evidentes
    dt_col = _infer_datetime_col(df_raw) or df_raw.columns[0]
    val_col = _infer_numeric_col(df_raw) or df_raw.columns[1]

    # Preprocesa la serie horaria
    try:
        df_price = preprocess_price(df_raw, dt_col, val_col, tz="America/Bogota")
    except Exception as e:
        st.error(f"Error en preprocesamiento: {e}")
        st.stop()

    st.session_state["df_price"] = df_price

    # Entrena el HMM y construye características/probabilidades
    with st.spinner("Ajustando HMM sobre retornos y construyendo características..."):
        hmm_model, df_feat, gamma_df = fit_hmm_and_features(df_price["precio"], n_states)

    # Entrena regresores globales y por régimen
    with st.spinner("Entrenando modelos de regresión por régimen..."):
        global_model, models, feature_cols, test_info = train_state_regressors(df_feat, n_states, horizon)

    # Hace persistencia con el session_state para evitar re-entrenar en cada recarga
    st.session_state["hmm_model"] = hmm_model
    st.session_state["df_feat"] = df_feat
    st.session_state["gamma_df"] = gamma_df
    st.session_state["global_model"] = global_model
    st.session_state["models"] = models
    st.session_state["feature_cols"] = feature_cols
    st.session_state["test_info"] = test_info

    st.success("Modelos entrenados correctamente.")


# Si no hay datos entrenados en session_state, se detiene la ejecución
if "df_price" not in st.session_state:
    st.info("Pulsa **'Traer datos de SIMEM y entrenar modelos'** para comenzar.")
    st.stop()


# Recupera objetos ya entrenados
df_price = st.session_state["df_price"]
hmm_model = st.session_state["hmm_model"]
df_feat = st.session_state["df_feat"]
gamma_df = st.session_state["gamma_df"]
global_model = st.session_state["global_model"]
models = st.session_state["models"]
feature_cols = st.session_state["feature_cols"]
X_test, y_test, test_index, test_len = st.session_state["test_info"]


st.subheader("📈 Serie de precio (preprocesada)")
st.write(
    f"Observaciones horarias: **{len(df_price)}** • "
    f"Rango: {df_price.index.min()} → {df_price.index.max()} • "
    f"CodigoVariable: **{codigo_variable}**"
)
st.plotly_chart(
    px.line(df_price, y="precio", title="Precio PBE horario (SIMEM • EC6945)"),
    use_container_width=True,
)


st.subheader("⛓️ Modelo Oculto de Markov (HMM)")

# Matriz de transición estimada por el HMM
P_hmm = hmm_model.transmat_
figP = go.Figure(
    data=go.Heatmap(
        z=P_hmm,
        x=[f"s{j}" for j in range(n_states)],
        y=[f"s{i}" for i in range(n_states)],
        colorscale="Blues",
        text=np.round(P_hmm, 3),
        hovertemplate="De s%{y} a s%{x}: %{z:.3f}<extra></extra>",
    )
)
figP.update_layout(title="Matriz de transición entre regímenes (HMM)")
st.plotly_chart(figP, use_container_width=True)

# Visualización de precio con color por estado Viterbi
df_states_plot = df_feat[["precio", "state"]].copy()
fig_states = px.scatter(
    df_states_plot,
    x=df_states_plot.index,
    y="precio",
    color="state",
    title="Precio histórico coloreado por régimen oculto (Viterbi, HMM)",
)
st.plotly_chart(fig_states, use_container_width=True)

# Probabilidades a posteriori del régimen (última semana)
gamma_tail = gamma_df.tail(7 * 24)
fig_gamma = px.area(
    gamma_tail,
    x=gamma_tail.index,
    y=gamma_tail.columns,
    title="Probabilidades de régimen (γ_t) — última semana (HMM)",
)
st.plotly_chart(fig_gamma, use_container_width=True)


st.subheader("🧪 Retroprueba de modelos (últimas horas)")

# Predicción del enfoque HMM+ML sobre el bloque de prueba
gamma_test = gamma_df.loc[test_index]
y_pred_hmm_ml: List[float] = []
for t in test_index:
    # x_row se toma de df_feat para asegurar consistencia de características
    x_row = df_feat.loc[t][feature_cols]
    # Probabilidades de regímenes en t
    probs_t = gamma_test.loc[t].values
    # Predicción ponderada por régimen
    y_hat_t = mixture_predict_row(x_row, probs_t, models, global_model, feature_cols)
    y_pred_hmm_ml.append(y_hat_t)

y_pred_hmm_ml = pd.Series(y_pred_hmm_ml, index=test_index)

metrics_rows = [
    {
        "Modelo": "HMM+ML",
        "MAE": mae(y_test, y_pred_hmm_ml),
        "RMSE": rmse(y_test, y_pred_hmm_ml),
        "MAPE (%)": mape(y_test, y_pred_hmm_ml),
        "sMAPE (%)": smape(y_test, y_pred_hmm_ml),
    }
]

# Predicción del enfoque MC-horaria (Modelo de Cadenas de Markov No Homogéneas por Hora)
price_aligned = df_price["precio"].loc[df_feat.index]
states_full, _ = discretize_states(price_aligned, mc_n_states, mc_disc_method)

n_total = len(price_aligned)
if test_len >= n_total:
    st.error("El horizonte de retroprueba es demasiado grande para la serie disponible.")
    st.stop()

train_len = n_total - test_len
prices_train = price_aligned.iloc[:train_len]
prices_test = price_aligned.iloc[train_len:]
idx_test_mc = prices_test.index
states_train = states_full.iloc[:train_len]

# Ajuste de matrices por hora y medias por estado usando solo entrenamiento
P_mc_train, state_means_mc_train = mc_hourly_fit(states_train, prices_train, mc_n_states)

preds_mc: List[float] = []
for idx in idx_test_mc:
    # El modelo usa la transición desde el estado previo (t-1) condicionado por la hora de t-1
    prev_idx = idx - pd.Timedelta(hours=1)
    if prev_idx in states_full.index:
        s_prev = int(states_full.loc[prev_idx])
    else:
        # Respaldo si falta el punto exacto
        s_prev = int(states_train.iloc[-1])

    h_prev = prev_idx.hour
    p_next = P_mc_train[h_prev, s_prev, :]

    # Valor esperado del precio en t: E[Precio|S_t] usando medias por estado
    y_hat = float(np.dot(p_next, state_means_mc_train))
    preds_mc.append(y_hat)

y_pred_mc = pd.Series(preds_mc, index=idx_test_mc)
y_pred_mc = y_pred_mc.loc[test_index]

metrics_rows.append(
    {
        "Modelo": f"MC-horaria (n={mc_n_states})",
        "MAE": mae(y_test, y_pred_mc),
        "RMSE": rmse(y_test, y_pred_mc),
        "MAPE (%)": mape(y_test, y_pred_mc),
        "sMAPE (%)": smape(y_test, y_pred_mc),
    }
)

st.dataframe(pd.DataFrame(metrics_rows).set_index("Modelo"), use_container_width=True)

# Gráfico comparativo de la prueba en las últimas horas
df_bt_plot = pd.DataFrame(
    {
        "real": y_test,
        "HMM+ML": y_pred_hmm_ml,
        "MC-horaria": y_pred_mc,
    },
    index=test_index,
)

fig_bt = go.Figure()

# Parte de entrenamiento para contexto visual
hist_idx = df_feat.index[:-test_len]
fig_bt.add_trace(
    go.Scatter(
        x=hist_idx,
        y=df_feat.loc[hist_idx, "precio"],
        name="Histórico (entrenamiento)",
        line=dict(width=1),
    )
)

fig_bt.add_trace(
    go.Scatter(
        x=df_bt_plot.index,
        y=df_bt_plot["real"],
        name="Real (prueba)",
        line=dict(width=2),
    )
)

fig_bt.add_trace(
    go.Scatter(
        x=df_bt_plot.index,
        y=df_bt_plot["HMM+ML"],
        name="Predicho HMM+ML",
        line=dict(width=2, dash="dot"),
    )
)

fig_bt.add_trace(
    go.Scatter(
        x=df_bt_plot.index,
        y=df_bt_plot["MC-horaria"],
        name="Predicho MC-horaria",
        line=dict(width=2, dash="dash"),
    )
)

fig_bt.update_layout(title="Retroprueba: real vs predicho (HMM+ML vs MC-horaria)")
st.plotly_chart(fig_bt, use_container_width=True)


st.subheader("🔮 Pronóstico del día siguiente (24 horas)")

# Pronóstico multi-paso HMM+ML
price_series_ext = df_price["precio"].copy()

# Se usa el último tiempo disponible en df_feat para asegurar disponibilidad de rezagos
last_time = df_feat.index[-1]

# Distribución inicial de regímenes para el pronóstico: gamma del último punto
p_state = gamma_df.iloc[-1].values.copy()

future_times: List[pd.Timestamp] = []
future_preds_hmm_ml: List[float] = []

for h in range(1, DEFAULT_HORIZON + 1):
    t = last_time + pd.Timedelta(hours=h)

    # Características construidas con histórico extendido
    feat_row = build_future_feature_row(price_series_ext, t)
    feat_row = feat_row.reindex(feature_cols)

    # Relleno defensivo ante nulos (puede ocurrir en bordes muy tempranos)
    feat_row = feat_row.fillna(method="ffill").fillna(method="bfill")

    y_hat_t = mixture_predict_row(feat_row, p_state, models, global_model, feature_cols)

    # Importante: se inserta la predicción en la serie extendida para soportar multi-paso
    price_series_ext.loc[t] = y_hat_t
    future_times.append(t)
    future_preds_hmm_ml.append(y_hat_t)

    # Actualizar distribución de regímenes: p_{t+1} = p_t · P
    p_state = p_state @ P_hmm

future_index = pd.DatetimeIndex(future_times)
df_future = pd.DataFrame({"HMM+ML": future_preds_hmm_ml}, index=future_index)

# Pronóstico multi-paso MC-horaria
states_fore, _ = discretize_states(df_price["precio"], mc_n_states, mc_disc_method)
P_mc_fore, state_means_mc_fore = mc_hourly_fit(states_fore, df_price["precio"], mc_n_states)

last_state_mc = int(states_fore.iloc[-1])
p_state_mc = np.zeros(mc_n_states, dtype=float)
p_state_mc[last_state_mc] = 1.0

future_preds_mc: List[float] = []
cur_time = df_price.index[-1]
for h in range(1, DEFAULT_HORIZON + 1):
    # La transición depende de la hora del instante previo
    prev_time = cur_time + pd.Timedelta(hours=h - 1)
    hour_prev = prev_time.hour

    # Distribución de estados para el siguiente paso
    p_state_mc = p_state_mc @ P_mc_fore[hour_prev]

    # Valor esperado del precio para el paso actual
    y_hat_mc = float(np.dot(p_state_mc, state_means_mc_fore))
    future_preds_mc.append(y_hat_mc)

df_future["MC-horaria"] = future_preds_mc


# Intenta traer el precio real del día siguiente, si ya está publicado
real_nextday: Optional[pd.DataFrame] = None
try:
    next_start = future_index[0].strftime("%Y-%m-%d")
    next_end = future_index[-1].strftime("%Y-%m-%d")
    df_raw_next = simem_fetch_ec6945(next_start, next_end)

    if not df_raw_next.empty:
        if "CodigoVariable" in df_raw_next.columns:
            df_raw_next = df_raw_next[df_raw_next["CodigoVariable"] == codigo_variable].copy()

        dt_col_next = _infer_datetime_col(df_raw_next) or df_raw_next.columns[0]
        val_col_next = _infer_numeric_col(df_raw_next) or df_raw_next.columns[1]
        df_price_next = preprocess_price(df_raw_next, dt_col_next, val_col_next, tz="America/Bogota")

        # Filtrar exactamente las horas pronosticadas
        real_nextday = df_price_next.loc[df_price_next.index.intersection(future_index)]
        if real_nextday.empty:
            real_nextday = None
except Exception as e:
    st.info(f"No se pudo recuperar el precio real del día siguiente desde SIMEM: {e}")


# Gráfico del pronóstico y del precio real (si existe)
lookback = min(len(df_price), 72)
fig_future = go.Figure()
fig_future.add_trace(
    go.Scatter(
        x=df_price.index[-lookback:],
        y=df_price["precio"].iloc[-lookback:],
        name="Histórico reciente",
    )
)
fig_future.add_trace(
    go.Scatter(
        x=df_future.index,
        y=df_future["HMM+ML"],
        name="Pronóstico HMM+ML",
    )
)
fig_future.add_trace(
    go.Scatter(
        x=df_future.index,
        y=df_future["MC-horaria"],
        name="Pronóstico MC-horaria",
    )
)

if real_nextday is not None:
    fig_future.add_trace(
        go.Scatter(
            x=real_nextday.index,
            y=real_nextday["precio"],
            name="Precio real día siguiente",
            line=dict(width=3, dash="dot"),
        )
    )

fig_future.update_layout(title="Pronóstico del día siguiente (24h) • HMM+ML vs MC-horaria")
st.plotly_chart(fig_future, use_container_width=True)


st.download_button(
    "💾 Descargar pronóstico futuro (CSV)",
    df_future.to_csv().encode("utf-8"),
    file_name=f"pronostico_hmm_mc_horaria_ec6945_{codigo_variable.lower()}.csv",
    mime="text/csv",
)
