# Modelado y Predicción del Precio de Bolsa de Energía en Colombia (PBE)

Este repositorio contiene una **aplicación en Streamlit** para pronóstico **horario** del **Precio de Bolsa de Energía (PBE)** en Colombia, utilizando datos históricos de **SIMEM** (dataset **EC6945**) y comparando dos enfoques:

- **Modelo 1 — HMM + ML:** Modelo Oculto de Markov (HMM) gaussiano sobre **retornos logarítmicos** para inferir regímenes, combinado con una **mezcla de expertos** basada en modelos de **boosting**.
- **Modelo 2 — MC-horaria:** **Modelo de Cadenas de Markov No Homogéneas por Hora** (MC-horaria) sobre **estados discretos** del precio.

> Nota: El informe del proyecto (LaTeX) está en `main.tex`.

---

## ¿Qué hace la aplicación?

La app descarga los datos, los preprocesa a frecuencia horaria y permite:

- Visualizar la serie horaria del precio.
- Inspeccionar el HMM: **matriz de transición**, **regímenes (Viterbi)** y **probabilidades a posteriori** por régimen.
- Ejecutar una **retroprueba** en las últimas horas del rango (con métricas MAE/RMSE/MAPE/sMAPE).
- Generar un **pronóstico de las próximas 24 horas** y, si ya existen datos publicados, **superponer el precio real del día siguiente** para comparar.
- Exportar el pronóstico a **CSV** desde la propia interfaz.

---

## Descripción de los modelos

### Modelo 1: HMM + ML (mezcla de expertos)

1. **HMM gaussiano sobre retornos logarítmicos**
   - Se trabaja con retornos (en vez de precio) para mejorar estabilidad.
   - Se obtiene:
     - Estados más probables por tiempo (Viterbi).
     - Probabilidades a posteriori por estado (γ_t).

2. **Tabla de características (features) para regresión**
   - Variables temporales: hora y día de la semana.
   - Codificación cíclica: seno/coseno para hora y día.
   - Rezagos: 1h, 2h, 24h.
   - Promedios móviles: 24h y 168h.

3. **Mezcla de expertos con boosting**
   - Se entrena un modelo global (respaldo).
   - Se entrena un modelo por régimen si hay suficientes muestras.
   - La predicción final es una combinación ponderada por γ_t:
     - ŷ_t = Σ_k p_k(t) · f_k(x_t)

---

### Modelo 2: MC-horaria (Cadenas de Markov No Homogéneas por Hora)

1. **Discretización del precio a estados**
   - Opción `quantiles`: cortes por cuantiles (robusto a outliers).
   - Opción `kmeans`: clustering 1D y reordenamiento por media.

2. **Matrices de transición por hora**
   - Para cada hora h, se estima una matriz P^(h):
     - P^(h)[i, j] = P(S_{t+1}=j | S_t=i, hora(t)=h)
   - Se usa **suavizado de Laplace** para evitar probabilidades nulas.

3. **Mapeo de estados a valores**
   - Se calcula la media del precio por estado y se usa el valor esperado:
     - E[Precio_t] = Σ_j P(S_t=j) · mean_price(j)

---

## Requisitos

- Python recomendado: **3.10+**
- Dependencias: ver `requirements.txt`.

---

## Instalación

```bash
# 1) Clonar el repositorio
git clone <repo>
cd <repo>

# 2) Crear y activar entorno virtual
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

# 3) Instalar dependencias
pip install -r requirements.txt
```

---

## Ejecución

```bash
streamlit run app.py
```

---

## Uso dentro de la app

1. En la barra lateral, selecciona:
   - **Fecha inicio / Fecha fin**
   - **CodigoVariable**: `PB_Nal`, `PB_Int`, `PB_Tie`
   - Parámetros del modelo:
     - `Regímenes HMM` (2–5)
     - `Estados MC-horaria` (3–10)
     - Método de discretización (`quantiles` o `kmeans`)
     - Horizonte de retroprueba (12–48 horas)

2. Pulsa **“Traer datos de SIMEM y entrenar modelos”**.

3. Revisa:
   - La serie histórica
   - Componentes del HMM (transiciones, regímenes y γ_t)
   - Retroprueba (métricas comparativas)
   - Pronóstico 24h (y “real del día siguiente” si existe)

4. Descarga el CSV del pronóstico desde el botón **“Descargar pronóstico futuro (CSV)”**.

---

## Estructura del repositorio

- `app.py`  
  Aplicación principal en Streamlit: descarga, preprocesamiento, entrenamiento, visualización y pronóstico.

- `requirements.txt`  
  Dependencias mínimas para ejecutar la app.

- `main.tex`  
  Informe LaTeX del proyecto (marco teórico, metodología, resultados y conclusiones).

---

## Consideraciones y notas

- La descarga depende de disponibilidad de **SIMEM / pydataxm** y conectividad a internet.
- Si el rango no devuelve registros:
  - Ajusta las fechas o cambia `CodigoVariable`.
- En discretización por cuantiles, si hay muchos valores repetidos, el número efectivo de estados puede reducirse.

---

## Autoría

- **Alejandro Higuera Castro**  
  Informe Final del Proyecto (II-2025) — Curso/asesoría: **Prof. Freddy Rolando Hernández Romero**.

## Nota

- Parte de la documentación y la elaboración del informe se realizó con ayuda de modelos de IA como ChatGPT.
