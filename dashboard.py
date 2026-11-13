import json
import json
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
PREDICTIONS_H3_PATH = OUTPUT_DIR / "predicciones_test_comparativa.csv"
HIST_H3_PATH = OUTPUT_DIR / "serie_historica_real_completa.parquet"
METRICS_H3_PATH = OUTPUT_DIR / "metricas_hito3_comparativa.csv"
METRICS_VAL_H3_PATH = OUTPUT_DIR / "metricas_hito3_validation.csv"
DIST_METRICS_H3_PATH = OUTPUT_DIR / "metricas_hito3_por_distribuidora.csv"
TARIFF_METRICS_H3_PATH = OUTPUT_DIR / "metricas_hito3_por_tarifa.csv"
MANIFEST_H3_PATH = OUTPUT_DIR / "hito3_run_manifest.json"

@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_data
def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)

@st.cache_data
def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)

@st.cache_data
def load_predictions(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        dtype={"DISTRIBUIDORA": "string", "UBIGEO": "string", "TARIFA": "string"}
    )


@st.cache_data(ttl=60, show_spinner=False)
def load_hito3_from_postgres() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Recupera exclusivamente resultados oficiales Hito 3 desde PostgreSQL local."""
    from psycopg.rows import dict_row
    from scripts.db_utils import connect_db

    with connect_db(row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM model_runs ORDER BY created_at DESC LIMIT 1")
            run = cur.fetchone()
            if run is None:
                raise RuntimeError("PostgreSQL no contiene una corrida Hito 3 persistida.")
            run_id = run["run_id"]
            cur.execute(
                '''SELECT distribuidora AS "DISTRIBUIDORA", ubigeo AS "UBIGEO", tarifa AS "TARIFA",
                          periodo AS "PERIODO", target_periodo AS "TARGET_PERIODO",
                          target_real_mwh AS "TARGET_REAL", prediction_mwh AS "PRED_CAMPEON",
                          residual_mwh AS "RESIDUO", absolute_error_mwh AS "ERROR_ABSOLUTO",
                          percentage_error AS "ERROR_PORCENTUAL"
                   FROM predictions WHERE run_id = %s
                   ORDER BY target_periodo, distribuidora, ubigeo, tarifa''',
                (run_id,),
            )
            predictions = pd.DataFrame(cur.fetchall())
            cur.execute(
                """SELECT segment_type, segment_value, n_observations, mae, rmse, r2, wape, mean_residual
                   FROM segment_metrics WHERE run_id = %s""",
                (run_id,),
            )
            segments = cur.fetchall()

    if predictions.empty:
        raise RuntimeError("La corrida PostgreSQL seleccionada no contiene predicciones TEST.")

    def segment_frame(segment_type: str) -> pd.DataFrame:
        rows = []
        for row in segments:
            if row["segment_type"] == segment_type:
                value = row["segment_value"]
                rows.append({
                    **value,
                    "OBSERVACIONES": row["n_observations"],
                    "MAE_MWh": row["mae"],
                    "RMSE_MWh": row["rmse"],
                    "R2": row["r2"],
                    "WAPE_pct": row["wape"],
                    "RESIDUO_MEDIO": row["mean_residual"],
                })
        return pd.DataFrame(rows)

    manifest = run["manifest"]
    return (
        predictions,
        pd.DataFrame(manifest.get("test_metrics", [])),
        pd.DataFrame(manifest.get("validation_metrics", [])),
        segment_frame("DISTRIBUIDORA"),
        segment_frame("TARIFA"),
        segment_frame("DISTRIBUIDORA_UBIGEO"),
        {
            **manifest,
            "_segmentos_por_periodo": segment_frame("TARGET_PERIODO").to_dict(orient="records"),
            "_segmentos_por_rango": segment_frame("RANGO_CONSUMO").to_dict(orient="records"),
        },
    )

def format_period(period: pd.Series) -> pd.Series:
    return pd.to_datetime(period.astype(str), format="%Y%m").dt.strftime("%Y-%m")

# Encabezado
st.title("Plataforma de Predicción de Demanda Eléctrica")
st.subheader("ENOSA & Hidrandina — Evaluación CRISP-DM")

# Selector de Hito / Vista en Sidebar
st.sidebar.title("Configuración")
hito_selection = st.sidebar.radio(
    "Selecciona la Fase de Modelado:",
    [
        "Hito 3: Modelos Avanzados (campeón por VALIDATION)",
        "Hito 2: Baseline Inicial (Decision Tree vs Referencia)",
    ],
    index=0
)

# Carga condicional según selección. Hito 3 prefiere PostgreSQL y conserva fallback local.
if "Hito 3" in hito_selection:
    try:
        (
            predictions, metrics_test, metrics_val, dist_metrics, tariff_metrics,
            ubigeo_metrics, h3_manifest,
        ) = load_hito3_from_postgres()
        history_df = None
        data_source_label = "PostgreSQL local"
    except Exception:
        predictions = load_predictions(PREDICTIONS_H3_PATH)
        history_df = load_parquet(HIST_H3_PATH) if HIST_H3_PATH.is_file() else None
        metrics_test = load_csv(METRICS_H3_PATH)
        metrics_val = load_csv(METRICS_VAL_H3_PATH) if METRICS_VAL_H3_PATH.is_file() else None
        dist_metrics = load_csv(DIST_METRICS_H3_PATH)
        tariff_metrics = load_csv(TARIFF_METRICS_H3_PATH)
        ubigeo_metrics = load_csv(OUTPUT_DIR / "hito3_errores_por_ubigeo.csv")
        h3_manifest = load_json(MANIFEST_H3_PATH) if MANIFEST_H3_PATH.is_file() else {}
        h3_manifest["_segmentos_por_periodo"] = load_csv(OUTPUT_DIR / "hito3_errores_por_mes.csv").to_dict(orient="records")
        h3_manifest["_segmentos_por_rango"] = load_csv(OUTPUT_DIR / "hito3_errores_por_rango_consumo.csv").to_dict(orient="records")
        data_source_label = "archivos locales"
    is_hito3 = True
else:
    predictions = load_predictions(PREDICTIONS_BASELINE_PATH)
    history_df = None
    metrics_test = load_csv(METRICS_BASELINE_PATH)
    metrics_val = None
    dist_metrics = load_csv(DIST_METRICS_BASELINE_PATH)
    tariff_metrics = load_csv(TARIFF_METRICS_BASELINE_PATH)
    ubigeo_metrics = pd.DataFrame()
    h3_manifest = {}
    data_source_label = "archivos locales"
    is_hito3 = False

st.sidebar.caption(f"Fuente de datos: {data_source_label}")

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
    st.markdown("### Desempeño Oficial en Conjunto TEST (2025-04 a 2025-07)")
    champion_name = h3_manifest.get("champion", {}).get("name", "LightGBM Avanzado (Hito 3)")
    champion_family = h3_manifest.get("champion", {}).get("family", "LightGBM")
    champion_pred_col = {
        "LightGBM": "PRED_LIGHTGBM",
        "XGBoost": "PRED_XGBOOST",
        "CatBoost": "PRED_CATBOOST",
    }.get(champion_family, "PRED_LIGHTGBM")
    if "PRED_CAMPEON" in predictions.columns:
        champion_pred_col = "PRED_CAMPEON"
    
    # KPIs principales del modelo campeón
    champ_row = metrics_test.loc[metrics_test["MODELO"] == champion_name].iloc[0]
    ref_row = metrics_test.loc[metrics_test["MODELO"].str.contains("Persistencia")].iloc[0]
    lgb_row = metrics_test.loc[metrics_test["MODELO"].str.contains("LightGBM")].iloc[0]
    cb_row = metrics_test.loc[metrics_test["MODELO"].str.contains("CatBoost")].iloc[0]
    
    kpis = st.columns(5)
    kpis[0].metric("MAE Persistencia (Ref.)", f"{ref_row['MAE_MWh']:.3f} MWh")
    kpis[1].metric(f"MAE {champion_family} (Campeón)", f"{champ_row['MAE_MWh']:.3f} MWh", delta=f"-{champ_row['MEJORA_MAE_pct']:.2f}% (Mejora)")
    kpis[2].metric("MAE LightGBM", f"{lgb_row['MAE_MWh']:.3f} MWh", delta=f"-{lgb_row['MEJORA_MAE_pct']:.2f}%")
    kpis[3].metric("MAE CatBoost", f"{cb_row['MAE_MWh']:.3f} MWh", delta=f"-{cb_row['MEJORA_MAE_pct']:.2f}%")
    kpis[4].metric(f"WAPE {champion_family}", f"{champ_row['WAPE_pct']:.2f}%", delta=f"-{champ_row['MEJORA_MAE_pct']:.2f}% MAE vs Ref")
    st.caption(f"Campeón oficial: {champion_name}, seleccionado exclusivamente por MAE en VALIDATION. Esta vista contiene solo TEST; FUTURE_HOLDOUT no se mezcla ni participa en la selección.")
    
    # Tabla comparativa completa
    st.markdown("#### Matriz Comparativa Oficial (TEST)")
    tab1, tab2, tab3 = st.tabs([
        "Evaluación TEST (Fuera de muestra)",
        "Selección Intermedia en VALIDATION",
        "Alcance metodológico"
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
        st.success(f" {champion_name} fue fijado antes de TEST por su MAE de VALIDATION. TEST se usa únicamente como evaluación final fuera de muestra.")
        
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
        st.info("La corrida oficial Hito 3 compara persistencia, Decision Tree y tres modelos avanzados con ajuste temporal dentro de TRAIN. No se usan comparativas históricas de arquitecturas para seleccionar el campeón ni para reportar métricas oficiales.")

    # Gráfico de Consumo Temporal: Empalme Histórico + Pronóstico Test
    st.markdown("###  Trayectoria Temporal: Demanda Histórica y Pronósticos (TEST Ciego)")
    
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
    temporal_columns = ["TARGET_REAL", champion_pred_col]
    for column in ["PRED_REFERENCIA", "PRED_XGBOOST", "PRED_LIGHTGBM", "PRED_CATBOOST"]:
        if column in filtered.columns and column not in temporal_columns:
            temporal_columns.append(column)
    temporal = filtered.groupby("TARGET_PERIODO", as_index=False)[temporal_columns].sum().sort_values("TARGET_PERIODO")
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

    # Modelo campeón seleccionado en VALIDATION
    y_champion = [last_hist_val] + temporal[champion_pred_col].tolist() if not hist_series.empty else temporal[champion_pred_col].tolist()
    fig_temp.add_trace(go.Scatter(
        x=x_real,
        y=y_champion,
        mode="lines+markers",
        name=f"{champion_family} (Campeón)",
        line={"color": "#2563eb", "width": 3, "dash": "solid"},
        marker={"size": 7, "symbol": "diamond"}
    ))

    for family, column, color, dash in [
        ("LightGBM", "PRED_LIGHTGBM", "#10b981", "dot"),
        ("XGBoost", "PRED_XGBOOST", "#8b5cf6", "dash"),
        ("CatBoost", "PRED_CATBOOST", "#0ea5e9", "dashdot"),
    ]:
        if family != champion_family and column in temporal.columns:
            y_model = [last_hist_val] + temporal[column].tolist() if not hist_series.empty else temporal[column].tolist()
            fig_temp.add_trace(go.Scatter(
                x=x_real,
                y=y_model,
                mode="lines+markers",
                name=family,
                line={"color": color, "width": 2, "dash": dash},
                marker={"size": 6}
            ))

    # Persistencia (Referencia)
    if "PRED_REFERENCIA" in temporal.columns:
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
            annotation_text="Ventana TEST Fuera de Muestra",
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
            scatter_df["y_plot"] = np.log1p(np.clip(scatter_df[champion_pred_col], 0, None))
            x_title, y_title = "log1p(Consumo Real)", f"log1p({champion_family})"
        else:
            scatter_df["x_plot"] = scatter_df["TARGET_REAL"]
            scatter_df["y_plot"] = scatter_df[champion_pred_col]
            x_title, y_title = "Consumo Real (MWh)", f"Predicción {champion_family} (MWh)"
            
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
        st.markdown(f"#### Distribución de Residuos ({champion_family})")
        residuos = filtered["TARGET_REAL"] - filtered[champion_pred_col]
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
    st.markdown("### Desglose Analítico por Segmentos")
    col_d, col_t = st.columns(2)
    with col_d:
        st.markdown("#### Desempeño por Distribuidora")
        st.dataframe(dist_metrics, width="stretch", hide_index=True)
    with col_t:
        st.markdown("#### Desempeño por Tarifa (Top 8)")
        st.dataframe(tariff_metrics, width="stretch", hide_index=True)
    with st.expander("Métricas por UBIGEO, periodo y rango de consumo"):
        st.markdown("#### Por UBIGEO")
        st.dataframe(ubigeo_metrics, width="stretch", hide_index=True)
        period_metrics = pd.DataFrame(h3_manifest.get("_segmentos_por_periodo", []))
        range_metrics = pd.DataFrame(h3_manifest.get("_segmentos_por_rango", []))
        left_metrics, right_metrics = st.columns(2)
        with left_metrics:
            st.markdown("#### Por TARGET_PERIODO")
            st.dataframe(period_metrics, width="stretch", hide_index=True)
        with right_metrics:
            st.markdown("#### Por rango de consumo")
            st.dataframe(range_metrics, width="stretch", hide_index=True)

    # Inspector puntual de predicción
    st.markdown("### Inspector Detallado de Observaciones")
    example_idx = st.selectbox(
        "Seleccionar serie para auditar:",
        filtered.index.tolist(),
        format_func=lambda i: f"{filtered.loc[i, 'DISTRIBUIDORA']} · UBIGEO {filtered.loc[i, 'UBIGEO']} · {filtered.loc[i, 'TARIFA']} · Periodo {filtered.loc[i, 'TARGET_PERIODO']}"
    )
    ex = filtered.loc[example_idx]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Consumo Real", f"{ex['TARGET_REAL']:.3f} MWh")
    if "PRED_REFERENCIA" in ex.index:
        c2.metric("Referencia (Mes anterior)", f"{ex['PRED_REFERENCIA']:.3f} MWh")
    else:
        c2.metric("Referencia", "No almacenada")
    c3.metric(f"{champion_family} (Campeón)", f"{ex[champion_pred_col]:.3f} MWh", delta=f"{ex[champion_pred_col] - ex['TARGET_REAL']:+.3f}")
    comparison_options = [("XGBoost", "PRED_XGBOOST"), ("LightGBM", "PRED_LIGHTGBM"), ("CatBoost", "PRED_CATBOOST")]
    comparison = next(((name, column) for name, column in comparison_options if column in ex.index and column != champion_pred_col), None)
    if comparison:
        comparison_name, comparison_col = comparison
        c4.metric(comparison_name, f"{ex[comparison_col]:.3f} MWh", delta=f"{ex[comparison_col] - ex['TARGET_REAL']:+.3f}")
    else:
        c4.metric("Modelo alternativo", "No almacenado")

# ==================== VISTA HITO 2 ====================
else:
    st.markdown("### Baseline Inicial — Decision Tree (Hito 2)")
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
