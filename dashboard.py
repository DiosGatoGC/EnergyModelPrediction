from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Predicción del consumo eléctrico — CRISP-DM",
    page_icon="⚡",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "reports" / "model_outputs"
MODELS_DIR = ROOT / "models"

# Rutas Hito 2
PREDICTIONS_BASELINE_PATH = OUTPUT_DIR / "predicciones_test.csv"
METRICS_BASELINE_PATH = OUTPUT_DIR / "metricas_baseline.csv"
DIST_METRICS_BASELINE_PATH = OUTPUT_DIR / "metricas_por_distribuidora.csv"
TARIFF_METRICS_BASELINE_PATH = OUTPUT_DIR / "metricas_por_tarifa.csv"

# Rutas Hito 3
PREDICTIONS_H3_PATH = OUTPUT_DIR / "predicciones_evaluacion_completa_2026.csv"
HIST_H3_PATH = OUTPUT_DIR / "serie_historica_real_completa.parquet"
METRICS_H3_PATH = OUTPUT_DIR / "metricas_hito3_comparativa.csv"
METRICS_VAL_H3_PATH = OUTPUT_DIR / "metricas_hito3_validation.csv"
DIST_METRICS_H3_PATH = OUTPUT_DIR / "metricas_hito3_por_distribuidora.csv"
TARIFF_METRICS_H3_PATH = OUTPUT_DIR / "metricas_hito3_por_tarifa.csv"
ARCH_STRATEGIES_PATH = OUTPUT_DIR / "comparativa_estrategias_arquitectura.csv"

@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_data
def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)

@st.cache_data
def load_predictions(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        dtype={"DISTRIBUIDORA": "string", "UBIGEO": "string", "TARIFA": "string"}
    )

def format_period(period: pd.Series) -> pd.Series:
    return pd.to_datetime(period.astype(str), format="%Y%m").dt.strftime("%Y-%m")

# Encabezado
st.title("⚡ Plataforma de Predicción de Demanda Eléctrica")
st.subheader("ENOSA & Hidrandina — Evaluación CRISP-DM (Hito 2 vs Hito 3)")

# Selector de Hito / Vista en Sidebar
st.sidebar.title("Configuración")
hito_selection = st.sidebar.radio(
    "Selecciona la Fase de Modelado:",
    [
        "🚀 Hito 3: Modelos Avanzados (XGBoost, LightGBM, CatBoost)",
        "📊 Hito 2: Baseline Inicial (Decision Tree vs Referencia)",
    ],
    index=0
)

# Carga condicional según selección
if "Hito 3" in hito_selection and PREDICTIONS_H3_PATH.is_file():
    predictions = load_predictions(PREDICTIONS_H3_PATH)
    history_df = load_parquet(HIST_H3_PATH) if HIST_H3_PATH.is_file() else None
    metrics_test = load_csv(METRICS_H3_PATH)
    metrics_val = load_csv(METRICS_VAL_H3_PATH) if METRICS_VAL_H3_PATH.is_file() else None
    dist_metrics = load_csv(DIST_METRICS_H3_PATH)
    tariff_metrics = load_csv(TARIFF_METRICS_H3_PATH)
    is_hito3 = True
else:
    predictions = load_predictions(PREDICTIONS_BASELINE_PATH)
    history_df = None
    metrics_test = load_csv(METRICS_BASELINE_PATH)
    metrics_val = None
    dist_metrics = load_csv(DIST_METRICS_BASELINE_PATH)
    tariff_metrics = load_csv(TARIFF_METRICS_BASELINE_PATH)
    is_hito3 = False

# Filtros encadenados
st.sidebar.markdown("---")
st.sidebar.subheader("Filtros Territoriales y Tarifarios")

dist_options = ["Todas"] + sorted(predictions["DISTRIBUIDORA"].dropna().unique().tolist())
selected_dist = st.sidebar.selectbox("Distribuidora", dist_options)

filter_base = predictions
if selected_dist != "Todas":
    filter_base = filter_base.loc[filter_base["DISTRIBUIDORA"] == selected_dist]

ubigeo_options = ["Todos"] + sorted(filter_base["UBIGEO"].dropna().astype(str).unique().tolist())
selected_ubigeo = st.sidebar.selectbox("UBIGEO (Distrito)", ubigeo_options)
if selected_ubigeo != "Todos":
    filter_base = filter_base.loc[filter_base["UBIGEO"].astype(str) == selected_ubigeo]

tariff_options = ["Todas"] + sorted(filter_base["TARIFA"].dropna().unique().tolist())
selected_tariff = st.sidebar.selectbox("Tarifa Eléctrica", tariff_options)
if selected_tariff != "Todas":
    filter_base = filter_base.loc[filter_base["TARIFA"] == selected_tariff]

filtered = filter_base
st.sidebar.metric("Observaciones seleccionadas", f"{len(filtered):,}")

if filtered.empty:
    st.warning("No existen registros para la combinación de filtros seleccionada.")
    st.stop()

# ==================== VISTA HITO 3 ====================
if is_hito3:
    st.markdown("### 🏆 Desempeño Oficial en Conjunto TEST (2025-04 a 2025-07)")
    
    # KPIs principales del modelo campeón
    champ_row = metrics_test.loc[metrics_test["MODELO"].str.contains("XGBoost")].iloc[0]
    ref_row = metrics_test.loc[metrics_test["MODELO"].str.contains("Persistencia")].iloc[0]
    lgb_row = metrics_test.loc[metrics_test["MODELO"].str.contains("LightGBM")].iloc[0]
    cb_row = metrics_test.loc[metrics_test["MODELO"].str.contains("CatBoost")].iloc[0]
    
    kpis = st.columns(5)
    kpis[0].metric("MAE Persistencia (Ref.)", f"{ref_row['MAE_MWh']:.3f} MWh")
    kpis[1].metric("MAE XGBoost (Campeón)", f"{champ_row['MAE_MWh']:.3f} MWh", delta=f"-{champ_row['MEJORA_MAE_pct']:.2f}% (Mejora)")
    kpis[2].metric("MAE LightGBM", f"{lgb_row['MAE_MWh']:.3f} MWh", delta=f"-{lgb_row['MEJORA_MAE_pct']:.2f}%")
    kpis[3].metric("MAE CatBoost", f"{cb_row['MAE_MWh']:.3f} MWh", delta=f"-{cb_row['MEJORA_MAE_pct']:.2f}%")
    kpis[4].metric("WAPE XGBoost", f"{champ_row['WAPE_pct']:.2f}%", delta="-0.78% vs Ref")
    
    # Tabla comparativa completa
    st.markdown("#### Matriz Comparativa Oficial (TEST)")
    tab1, tab2, tab3 = st.tabs([
        "Evaluación TEST (Fuera de muestra)",
        "Selección Intermedia en VALIDATION",
        "🔬 Estudio de Arquitecturas (Global vs Distribuidora vs Tipo de Usuario)"
    ])
    
    with tab1:
        st.dataframe(
            metrics_test.style.format({
                "MAE_MWh": "{:.4f}",
                "RMSE_MWh": "{:.4f}",
                "R2": "{:.4f}",
                "WAPE_pct": "{:.2f}%",
                "MEJORA_MAE_pct": "{:+.2f}%",
            }),
            width="stretch",
            hide_index=True
        )
        st.success("✅ Los tres modelos avanzados (XGBoost, LightGBM, CatBoost) superaron el criterio de éxito del 10% de mejora sobre la persistencia.")
        
    with tab2:
        if metrics_val is not None:
            st.dataframe(
                metrics_val.style.format({
                    "MAE_MWh": "{:.4f}",
                    "RMSE_MWh": "{:.4f}",
                    "R2": "{:.4f}",
                    "WAPE_pct": "{:.2f}%",
                    "MEJORA_MAE_pct": "{:+.2f}%",
                }),
                width="stretch",
                hide_index=True
            )
            st.info("El conjunto VALIDATION (2025-01 a 2025-03) se utilizó formalmente como árbitro imparcial para seleccionar el modelo campeón antes del reentrenamiento final.")

    with tab3:
        full_mat_path = OUTPUT_DIR / "matriz_completa_estrategias_algoritmos.csv"
        if full_mat_path.is_file():
            df_full_mat = load_csv(full_mat_path)
            st.markdown("##### Benchmarking Completo: 4 Estrategias de Agrupamiento × 3 Algoritmos (TEST)")
            st.dataframe(
                df_full_mat.style.format({
                    "MAE_MWh": "{:.4f}",
                    "RMSE_MWh": "{:.4f}",
                    "R2": "{:.4f}",
                    "WAPE_pct": "{:.2f}%",
                    "Mejora_MAE_pct": "{:+.2f}%",
                }),
                width="stretch",
                hide_index=True
            )
            st.info("💡 **Conclusión del Benchmarking:** La segmentación por **Tipo de Usuario (BT Residencial vs MT Industrial)** con XGBoost logra el menor error (MAE 8.7727 MWh, RMSE 48.42 MWh). La sobre-fragmentación territorial (por distribuidora o en 4 modelos) degrada el desempeño por inanición de datos (data starvation).")
        elif ARCH_STRATEGIES_PATH.is_file():
            df_arch = load_csv(ARCH_STRATEGIES_PATH)
            st.dataframe(df_arch, width="stretch", hide_index=True)

    # Gráfico de Consumo Temporal: Empalme Histórico + Pronóstico Test
    st.markdown("### 📈 Trayectoria Temporal: Demanda Histórica y Pronósticos (TEST Ciego)")
    
    # Preparar histórico filtrado con los mismos filtros
    if history_df is not None:
        hist_filt = history_df.copy()
        if selected_dist != "Todas":
            hist_filt = hist_filt.loc[hist_filt["DISTRIBUIDORA"] == selected_dist]
        if selected_ubigeo != "Todos":
            hist_filt = hist_filt.loc[hist_filt["UBIGEO"].astype(str) == selected_ubigeo]
        if selected_tariff != "Todas":
            hist_filt = hist_filt.loc[hist_filt["TARIFA"] == selected_tariff]
            
        hist_series = (
            hist_filt.groupby("PERIODO", as_index=False)["consumo_total_mwh"]
            .sum()
            .sort_values("PERIODO")
        )
        hist_series["Periodo"] = format_period(hist_series["PERIODO"])
        hist_series = hist_series.rename(columns={"consumo_total_mwh": "Consumo Real"})
    else:
        hist_series = pd.DataFrame()

    # Preparar predicciones test
    temporal = (
        filtered.groupby("TARGET_PERIODO", as_index=False)[
            ["TARGET_REAL", "PRED_REFERENCIA", "PRED_XGBOOST", "PRED_LIGHTGBM", "PRED_CATBOOST"]
        ]
        .sum()
        .sort_values("TARGET_PERIODO")
    )
    temporal["Periodo"] = format_period(temporal["TARGET_PERIODO"])

    # Gráfico interactivo con Plotly Graph Objects
    fig_temp = go.Figure()

    # 1. Traza de Histórico (Entrenamiento + Validación)
    if not hist_series.empty:
        # Último punto histórico para conectar visualmente con Test
        last_hist_period = hist_series["Periodo"].iloc[-1]
        last_hist_val = hist_series["Consumo Real"].iloc[-1]
        
        fig_temp.add_trace(go.Scatter(
            x=hist_series["Periodo"],
            y=hist_series["Consumo Real"],
            mode="lines+markers",
            name="Histórico (Train/Val)",
            line={"color": "#4b5563", "width": 2.5},
            marker={"size": 5}
        ))

    # 2. Conexión y Trazas en TEST
    test_periods = temporal["Periodo"].tolist()
    
    # Consumo Real en TEST
    x_real = [last_hist_period] + test_periods if not hist_series.empty else test_periods
    y_real = [last_hist_val] + temporal["TARGET_REAL"].tolist() if not hist_series.empty else temporal["TARGET_REAL"].tolist()
    fig_temp.add_trace(go.Scatter(
        x=x_real,
        y=y_real,
        mode="lines+markers",
        name="Consumo Real (TEST)",
        line={"color": "#111827", "width": 3.5},
        marker={"size": 7, "symbol": "circle"}
    ))

    # XGBoost Campeón
    y_xgb = [last_hist_val] + temporal["PRED_XGBOOST"].tolist() if not hist_series.empty else temporal["PRED_XGBOOST"].tolist()
    fig_temp.add_trace(go.Scatter(
        x=x_real,
        y=y_xgb,
        mode="lines+markers",
        name="XGBoost (Campeón)",
        line={"color": "#2563eb", "width": 3, "dash": "solid"},
        marker={"size": 7, "symbol": "diamond"}
    ))

    # LightGBM
    y_lgb = [last_hist_val] + temporal["PRED_LIGHTGBM"].tolist() if not hist_series.empty else temporal["PRED_LIGHTGBM"].tolist()
    fig_temp.add_trace(go.Scatter(
        x=x_real,
        y=y_lgb,
        mode="lines+markers",
        name="LightGBM",
        line={"color": "#10b981", "width": 2, "dash": "dot"},
        marker={"size": 6}
    ))

    # CatBoost
    y_cb = [last_hist_val] + temporal["PRED_CATBOOST"].tolist() if not hist_series.empty else temporal["PRED_CATBOOST"].tolist()
    fig_temp.add_trace(go.Scatter(
        x=x_real,
        y=y_cb,
        mode="lines+markers",
        name="CatBoost",
        line={"color": "#8b5cf6", "width": 2, "dash": "dash"},
        marker={"size": 6}
    ))

    # Persistencia (Referencia)
    y_ref = [last_hist_val] + temporal["PRED_REFERENCIA"].tolist() if not hist_series.empty else temporal["PRED_REFERENCIA"].tolist()
    fig_temp.add_trace(go.Scatter(
        x=x_real,
        y=y_ref,
        mode="lines+markers",
        name="Persistencia y(t)",
        line={"color": "#f59e0b", "width": 2, "dash": "longdash"},
        marker={"size": 5}
    ))

    # Zona sombreada para TEST
    if test_periods:
        fig_temp.add_vrect(
            x0=test_periods[0],
            x1=test_periods[-1],
            fillcolor="#3b82f6",
            opacity=0.08,
            layer="below",
            line_width=1,
            line_dash="dash",
            line_color="#3b82f6",
            annotation_text="🎯 Ventana TEST Fuera de Muestra",
            annotation_position="top left",
        )

    fig_temp.update_layout(
        xaxis_title="Periodo Mensual",
        yaxis_title="Consumo de Energía (MWh)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_temp, width="stretch")

    # Gráficos de Diagnóstico: Scatter y Residuos
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("#### Real vs Predicho (Escala log1p)")
        use_log = st.checkbox("Escala log1p (recomendada por dispersión)", value=True)
        scatter_df = filtered.copy()
        if use_log:
            scatter_df["x_plot"] = np.log1p(np.clip(scatter_df["TARGET_REAL"], 0, None))
            scatter_df["y_plot"] = np.log1p(np.clip(scatter_df["PRED_XGBOOST"], 0, None))
            x_title, y_title = "log1p(Consumo Real)", "log1p(XGBoost)"
        else:
            scatter_df["x_plot"] = scatter_df["TARGET_REAL"]
            scatter_df["y_plot"] = scatter_df["PRED_XGBOOST"]
            x_title, y_title = "Consumo Real (MWh)", "Predicción XGBoost (MWh)"
            
        fig_scatter = px.scatter(
            scatter_df,
            x="x_plot",
            y="y_plot",
            color="DISTRIBUIDORA",
            hover_data=["UBIGEO", "TARIFA"],
            opacity=0.45,
            labels={"x_plot": x_title, "y_plot": y_title},
            color_discrete_map={"ENOSA": "#2563eb", "HIDRANDINA": "#16a34a"}
        )
        lim = float(max(scatter_df["x_plot"].max(), scatter_df["y_plot"].max()))
        fig_scatter.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines", name="Ideal y = x", line={"color": "#dc2626", "dash": "dash"}))
        fig_scatter.update_layout(legend_title_text="")
        st.plotly_chart(fig_scatter, width="stretch")

    with col_right:
        st.markdown("#### Distribución de Residuos (XGBoost)")
        residuos = filtered["TARGET_REAL"] - filtered["PRED_XGBOOST"]
        res_clipped = np.clip(residuos, -100, 100)
        fig_res = px.histogram(
            x=res_clipped,
            nbins=50,
            color_discrete_sequence=["#3b82f6"],
            labels={"x": "Error Residual (MWh)"},
            title="Distribución centrada en 0 (sin sesgo sistemático)"
        )
        fig_res.add_vline(x=0, line_dash="dash", line_color="#dc2626")
        st.plotly_chart(fig_res, width="stretch")

    # Desgloses por Distribuidora y Tarifa
    st.markdown("### 📊 Desglose Analítico por Segmentos")
    col_d, col_t = st.columns(2)
    with col_d:
        st.markdown("#### Desempeño por Distribuidora")
        st.dataframe(dist_metrics, width="stretch", hide_index=True)
    with col_t:
        st.markdown("#### Desempeño por Tarifa (Top 8)")
        st.dataframe(tariff_metrics, width="stretch", hide_index=True)

    # Inspector puntual de predicción
    st.markdown("### 🔍 Inspector Detallado de Observaciones")
    example_idx = st.selectbox(
        "Seleccionar serie para auditar:",
        filtered.index.tolist(),
        format_func=lambda i: f"{filtered.loc[i, 'DISTRIBUIDORA']} · UBIGEO {filtered.loc[i, 'UBIGEO']} · {filtered.loc[i, 'TARIFA']} · Periodo {filtered.loc[i, 'TARGET_PERIODO']}"
    )
    ex = filtered.loc[example_idx]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Consumo Real", f"{ex['TARGET_REAL']:.3f} MWh")
    c2.metric("Referencia (Mes anterior)", f"{ex['PRED_REFERENCIA']:.3f} MWh")
    c3.metric("XGBoost (Campeón)", f"{ex['PRED_XGBOOST']:.3f} MWh", delta=f"{ex['PRED_XGBOOST'] - ex['TARGET_REAL']:+.3f}")
    c4.metric("LightGBM", f"{ex['PRED_LIGHTGBM']:.3f} MWh", delta=f"{ex['PRED_LIGHTGBM'] - ex['TARGET_REAL']:+.3f}")

# ==================== VISTA HITO 2 ====================
else:
    st.markdown("### 📌 Baseline Inicial — Decision Tree (Hito 2)")
    test_metrics = metrics_test.loc[metrics_test["CONJUNTO"].astype(str).str.upper() == "TEST"]
    ref_row = test_metrics.loc[test_metrics["MODELO"] == "Referencia temporal"].iloc[0]
    tree_row = test_metrics.loc[test_metrics["MODELO"] == "Decision Tree"].iloc[0]
    
    kpis = st.columns(4)
    kpis[0].metric("MAE Referencia temporal", f"{ref_row['MAE']:.3f} MWh")
    kpis[1].metric("MAE Decision Tree", f"{tree_row['MAE']:.3f} MWh", delta=f"{tree_row['MEJORA_MAE_PCT']:.1f}%")
    kpis[2].metric("RMSE Decision Tree", f"{tree_row['RMSE']:.3f} MWh")
    kpis[3].metric("R² Decision Tree", f"{tree_row['R2']:.4f}")
    
    st.warning("En el Hito 2, el DecisionTreeRegressor no superó la referencia temporal (-24.74% en MAE), debido a la alta cardinalidad de UBIGEO y la aproximación escalonada sobre niveles absolutos.")
    st.dataframe(test_metrics, width="stretch", hide_index=True)
