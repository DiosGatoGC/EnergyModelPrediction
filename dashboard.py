from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Predicción del consumo eléctrico",
    page_icon="⚡",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "reports" / "model_outputs"
PREDICTIONS_PATH = OUTPUT_DIR / "predicciones_test.csv"
METRICS_PATH = OUTPUT_DIR / "metricas_baseline.csv"
DISTRIBUTOR_METRICS_PATH = OUTPUT_DIR / "metricas_por_distribuidora.csv"
TARIFF_METRICS_PATH = OUTPUT_DIR / "metricas_por_tarifa.csv"
SEARCH_RESULTS_PATH = OUTPUT_DIR / "resultados_validacion_arbol.csv"
MODEL_PATH = ROOT / "models" / "baseline_decision_tree.joblib"

REQUIRED_FILES = {
    "Predicciones de TEST": PREDICTIONS_PATH,
    "Métricas globales": METRICS_PATH,
    "Métricas por distribuidora": DISTRIBUTOR_METRICS_PATH,
    "Métricas por tarifa": TARIFF_METRICS_PATH,
    "Resultados de validación": SEARCH_RESULTS_PATH,
    "Pipeline del baseline": MODEL_PATH,
}

PREDICTION_COLUMNS = {
    "DISTRIBUIDORA",
    "UBIGEO",
    "TARIFA",
    "PERIODO",
    "TARGET_PERIODO",
    "TARGET_CONSUMO_MWH",
    "PRED_REFERENCIA",
    "PRED_DECISION_TREE",
    "ERROR_ABS_REFERENCIA",
    "ERROR_ABS_DECISION_TREE",
}
METRIC_COLUMNS = {"CONJUNTO", "MODELO", "MAE", "RMSE", "R2", "MEJORA_MAE_PCT"}
DISTRIBUTOR_METRIC_COLUMNS = {
    "DISTRIBUIDORA",
    "MODELO",
    "MAE",
    "RMSE",
    "R2",
    "MEJORA_MAE_PCT",
}
TARIFF_METRIC_COLUMNS = {
    "TARIFA",
    "cantidad_observaciones",
    "MAE_referencia",
    "MAE_decision_tree",
}


@st.cache_data
def load_predictions(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"DISTRIBUIDORA": "string", "UBIGEO": "string", "TARIFA": "string"})


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def validate_columns(frame: pd.DataFrame, expected: set[str], source_name: str) -> None:
    missing = sorted(expected - set(frame.columns))
    if missing:
        st.error(f"El archivo {source_name} no contiene las columnas esperadas: {', '.join(missing)}")
        st.stop()


def filtered_mae(frame: pd.DataFrame) -> tuple[float, float]:
    mae_reference = float(np.mean(np.abs(frame["TARGET_CONSUMO_MWH"] - frame["PRED_REFERENCIA"])))
    mae_tree = float(np.mean(np.abs(frame["TARGET_CONSUMO_MWH"] - frame["PRED_DECISION_TREE"])))
    return mae_reference, mae_tree


def format_period(period: pd.Series) -> pd.Series:
    return pd.to_datetime(period.astype(str), format="%Y%m").dt.strftime("%Y-%m")


missing_files = [f"{name}: {path}" for name, path in REQUIRED_FILES.items() if not path.is_file()]
if missing_files:
    st.error("Faltan artefactos necesarios para iniciar el dashboard:\n\n" + "\n".join(missing_files))
    st.stop()

try:
    predictions = load_predictions(PREDICTIONS_PATH)
    metrics = load_csv(METRICS_PATH)
    distributor_metrics = load_csv(DISTRIBUTOR_METRICS_PATH)
    tariff_metrics = load_csv(TARIFF_METRICS_PATH)
    search_results = load_csv(SEARCH_RESULTS_PATH)
except (OSError, ValueError, pd.errors.ParserError) as exc:
    st.error(f"No fue posible cargar los resultados del baseline: {exc}")
    st.stop()

validate_columns(predictions, PREDICTION_COLUMNS, PREDICTIONS_PATH.name)
validate_columns(metrics, METRIC_COLUMNS, METRICS_PATH.name)
validate_columns(distributor_metrics, DISTRIBUTOR_METRIC_COLUMNS, DISTRIBUTOR_METRICS_PATH.name)
validate_columns(tariff_metrics, TARIFF_METRIC_COLUMNS, TARIFF_METRICS_PATH.name)

test_metrics = metrics.loc[metrics["CONJUNTO"].astype(str).str.upper() == "TEST"].copy()
reference_rows = test_metrics.loc[test_metrics["MODELO"] == "Referencia temporal"]
tree_rows = test_metrics.loc[test_metrics["MODELO"] == "Decision Tree"]
if len(reference_rows) != 1 or len(tree_rows) != 1:
    st.error("metricas_baseline.csv debe contener una fila de TEST para cada modelo esperado.")
    st.stop()

reference_test = reference_rows.iloc[0]
tree_test = tree_rows.iloc[0]
selected_search = search_results.loc[search_results.get("seleccionado", False).astype(str).str.lower() == "true"]

st.title("⚡ Predicción del consumo eléctrico")
st.subheader("Baseline predictivo por distribuidora, UBIGEO y tarifa")
st.write(
    "Dashboard preliminar del Hito 2. Los resultados corresponden al conjunto TEST y "
    "comparan una referencia temporal con un árbol de decisión."
)

# Filtros encadenados: cada selector solo muestra valores compatibles con los anteriores.
st.sidebar.title("Filtros")
distributor_options = ["Todas"] + sorted(predictions["DISTRIBUIDORA"].dropna().unique().tolist())
selected_distributor = st.sidebar.selectbox("DISTRIBUIDORA", distributor_options)

filter_base = predictions
if selected_distributor != "Todas":
    filter_base = filter_base.loc[filter_base["DISTRIBUIDORA"] == selected_distributor]

ubigeo_options = ["Todos"] + sorted(filter_base["UBIGEO"].dropna().astype(str).unique().tolist())
selected_ubigeo = st.sidebar.selectbox("UBIGEO", ubigeo_options)
if selected_ubigeo != "Todos":
    filter_base = filter_base.loc[filter_base["UBIGEO"].astype(str) == selected_ubigeo]

tariff_options = ["Todas"] + sorted(filter_base["TARIFA"].dropna().unique().tolist())
selected_tariff = st.sidebar.selectbox("TARIFA", tariff_options)

filtered = filter_base
if selected_tariff != "Todas":
    filtered = filtered.loc[filtered["TARIFA"] == selected_tariff]

st.sidebar.metric("Observaciones seleccionadas", f"{len(filtered):,}")
active_filters = any(
    [selected_distributor != "Todas", selected_ubigeo != "Todos", selected_tariff != "Todas"]
)

if filtered.empty:
    st.warning("No existen observaciones para esta combinación.")
    st.stop()

st.markdown("### KPIs globales de TEST")
kpi_columns = st.columns(5)
kpi_columns[0].metric("MAE Referencia temporal", f"{reference_test['MAE']:.3f} MWh")
kpi_columns[1].metric("MAE Decision Tree", f"{tree_test['MAE']:.3f} MWh")
kpi_columns[2].metric("RMSE Decision Tree", f"{tree_test['RMSE']:.3f} MWh")
kpi_columns[3].metric("R² Decision Tree", f"{tree_test['R2']:.4f}")
kpi_columns[4].metric("Mejora MAE del árbol", f"{tree_test['MEJORA_MAE_PCT']:.1f}%")
st.caption("Una mejora negativa indica que el baseline ML tiene mayor MAE que la referencia temporal.")

st.markdown("## Comparación de modelos")
comparison_table = test_metrics[["MODELO", "MAE", "RMSE", "R2"]].copy()
comparison_table.columns = ["Modelo", "MAE", "RMSE", "R²"]
st.dataframe(
    comparison_table.style.format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "R²": "{:.4f}"}),
    width="stretch",
    hide_index=True,
)
if tree_test["MAE"] < reference_test["MAE"]:
    st.success("El Decision Tree reduce el error respecto a la referencia temporal.")
else:
    st.info(
        "La referencia temporal obtiene menor error que el Decision Tree. Esto indica que "
        "la persistencia mensual es un baseline fuerte para estos datos."
    )

st.markdown("## Consumo real vs predicho")
temporal = (
    filtered.groupby("TARGET_PERIODO", as_index=False)[
        ["TARGET_CONSUMO_MWH", "PRED_REFERENCIA", "PRED_DECISION_TREE"]
    ]
    .sum()
    .sort_values("TARGET_PERIODO")
)
temporal["Periodo objetivo"] = format_period(temporal["TARGET_PERIODO"])
temporal_long = temporal.melt(
    id_vars="Periodo objetivo",
    value_vars=["TARGET_CONSUMO_MWH", "PRED_REFERENCIA", "PRED_DECISION_TREE"],
    var_name="Serie",
    value_name="Consumo total (MWh)",
)
temporal_long["Serie"] = temporal_long["Serie"].map(
    {
        "TARGET_CONSUMO_MWH": "Real",
        "PRED_REFERENCIA": "Referencia temporal",
        "PRED_DECISION_TREE": "Baseline ML - Decision Tree",
    }
)
temporal_figure = px.line(
    temporal_long,
    x="Periodo objetivo",
    y="Consumo total (MWh)",
    color="Serie",
    markers=True,
    color_discrete_map={
        "Real": "#111827",
        "Referencia temporal": "#f59e0b",
        "Baseline ML - Decision Tree": "#2563eb",
    },
)
temporal_figure.update_layout(legend_title_text="", hovermode="x unified")
st.plotly_chart(temporal_figure, width="stretch")

st.markdown("## Relación entre consumo real y predicho")
use_log_scale = st.checkbox("Usar escala logarítmica")
scatter_data = filtered[["TARGET_CONSUMO_MWH", "PRED_DECISION_TREE", "DISTRIBUIDORA", "UBIGEO", "TARIFA"]].copy()
if use_log_scale:
    scatter_data["Real visual"] = np.log1p(np.clip(scatter_data["TARGET_CONSUMO_MWH"], 0, None))
    scatter_data["Predicción visual"] = np.log1p(np.clip(scatter_data["PRED_DECISION_TREE"], 0, None))
    x_label, y_label = "log1p(consumo real)", "log1p(consumo predicho)"
else:
    scatter_data["Real visual"] = scatter_data["TARGET_CONSUMO_MWH"]
    scatter_data["Predicción visual"] = scatter_data["PRED_DECISION_TREE"]
    x_label, y_label = "Consumo real (MWh)", "Consumo predicho (MWh)"

scatter_figure = px.scatter(
    scatter_data,
    x="Real visual",
    y="Predicción visual",
    color="DISTRIBUIDORA",
    hover_data=["UBIGEO", "TARIFA"],
    opacity=0.55,
    labels={"Real visual": x_label, "Predicción visual": y_label},
)
ideal_limit = float(max(scatter_data["Real visual"].max(), scatter_data["Predicción visual"].max()))
scatter_figure.add_trace(
    go.Scatter(
        x=[0, ideal_limit],
        y=[0, ideal_limit],
        mode="lines",
        name="Ideal y = x",
        line={"color": "#dc2626", "dash": "dash"},
    )
)
scatter_figure.update_layout(legend_title_text="")
st.plotly_chart(scatter_figure, width="stretch")
if use_log_scale:
    st.caption("La transformación log1p se usa únicamente para visualizar; las métricas permanecen en MWh.")

st.markdown("## Comparación del error")
if active_filters:
    mae_reference, mae_tree = filtered_mae(filtered)
    error_scope = "segmento filtrado"
else:
    mae_reference, mae_tree = float(reference_test["MAE"]), float(tree_test["MAE"])
    error_scope = "TEST global oficial"
mae_frame = pd.DataFrame(
    {
        "Modelo": ["Referencia temporal", "Baseline ML - Decision Tree"],
        "MAE (MWh)": [mae_reference, mae_tree],
    }
)
mae_figure = px.bar(
    mae_frame,
    x="Modelo",
    y="MAE (MWh)",
    color="Modelo",
    text_auto=".3f",
    color_discrete_map={"Referencia temporal": "#f59e0b", "Baseline ML - Decision Tree": "#2563eb"},
)
mae_figure.update_layout(showlegend=False, title=f"MAE — {error_scope}")
st.plotly_chart(mae_figure, width="stretch")

st.markdown("## Resultados por distribuidora")
display_distributor_metrics = distributor_metrics.copy()
display_distributor_metrics["MODELO"] = display_distributor_metrics["MODELO"].replace(
    {"Decision Tree": "Baseline ML - Decision Tree"}
)
st.dataframe(
    display_distributor_metrics.style.format(
        {"MAE": "{:.4f}", "RMSE": "{:.4f}", "R2": "{:.4f}", "MEJORA_MAE_PCT": "{:.2f}%"}
    ),
    width="stretch",
    hide_index=True,
)
distributor_figure = px.bar(
    display_distributor_metrics,
    x="DISTRIBUIDORA",
    y="MAE",
    color="MODELO",
    barmode="group",
    text_auto=".3f",
    labels={"MAE": "MAE (MWh)", "MODELO": "Modelo"},
    color_discrete_map={"Referencia temporal": "#f59e0b", "Baseline ML - Decision Tree": "#2563eb"},
)
st.plotly_chart(distributor_figure, width="stretch")

with st.expander("Resultados por tarifa"):
    tariff_display = tariff_metrics.sort_values("cantidad_observaciones", ascending=False).copy()
    tariff_display.columns = ["TARIFA", "Observaciones", "MAE referencia", "MAE Decision Tree"]
    st.dataframe(
        tariff_display.style.format({"MAE referencia": "{:.4f}", "MAE Decision Tree": "{:.4f}"}),
        width="stretch",
        hide_index=True,
    )

st.markdown("## Detalle de predicciones")
row_limit = st.select_slider("Cantidad máxima de observaciones", options=[50, 100, 200, 500], value=100)
detail_columns = [column for column in PREDICTION_COLUMNS if column in filtered.columns]
detail_columns = [
    "DISTRIBUIDORA",
    "UBIGEO",
    "TARIFA",
    "PERIODO",
    "TARGET_PERIODO",
    "TARGET_CONSUMO_MWH",
    "PRED_REFERENCIA",
    "PRED_DECISION_TREE",
    "ERROR_ABS_REFERENCIA",
    "ERROR_ABS_DECISION_TREE",
]
st.dataframe(filtered[detail_columns].head(row_limit), width="stretch", hide_index=True)

st.markdown("## Ejemplo de predicción")
example_options = filtered.index.tolist()

def example_label(index: int) -> str:
    row = filtered.loc[index]
    return (
        f"{row['DISTRIBUIDORA']} · UBIGEO {row['UBIGEO']} · {row['TARIFA']} · "
        f"objetivo {int(row['TARGET_PERIODO'])}"
    )

selected_example = st.selectbox("Selecciona una observación real de TEST", example_options, format_func=example_label)
example = filtered.loc[selected_example]
example_columns = st.columns(4)
example_columns[0].metric("Consumo real", f"{example['TARGET_CONSUMO_MWH']:.3f} MWh")
example_columns[1].metric("Predicción referencia", f"{example['PRED_REFERENCIA']:.3f} MWh")
example_columns[2].metric("Predicción Decision Tree", f"{example['PRED_DECISION_TREE']:.3f} MWh")
example_columns[3].metric("Error absoluto del árbol", f"{example['ERROR_ABS_DECISION_TREE']:.3f} MWh")
st.write(
    f"Para **{example['DISTRIBUIDORA']} / {example['UBIGEO']} / {example['TARIFA']}**, usando la "
    f"información del periodo **{int(example['PERIODO'])}**, el árbol estimó "
    f"**{example['PRED_DECISION_TREE']:.3f} MWh** para **{int(example['TARGET_PERIODO'])}**; "
    f"el valor real fue **{example['TARGET_CONSUMO_MWH']:.3f} MWh**. "
    f"Los errores absolutos fueron **{example['ERROR_ABS_REFERENCIA']:.3f} MWh** para la referencia "
    f"y **{example['ERROR_ABS_DECISION_TREE']:.3f} MWh** para el árbol."
)
st.caption("Ejemplo retrospectivo sobre TEST; no es una predicción de producción.")

with st.expander("Información del modelo"):
    selected_parameters = selected_search.iloc[0] if not selected_search.empty else None
    st.markdown(
        """
        **Tipo de problema:** Regresión  
        **Modelo:** DecisionTreeRegressor

        **Features**
        - DISTRIBUIDORA
        - UBIGEO
        - TARIFA
        - MES
        - consumo_total_mwh
        - lag_1
        - lag_2
        - lag_3

        **Métricas:** MAE, RMSE y R²

        La selección del modelo se realizó utilizando Validation. Test se mantuvo separado hasta la evaluación final.

        FUTURE_HOLDOUT no fue utilizado para entrenamiento, selección ni evaluación del Hito 2.
        """
    )
    if selected_parameters is not None:
        st.markdown(
            f"""
            **Hiperparámetros seleccionados**
            - max_depth = {selected_parameters['max_depth']}
            - min_samples_leaf = {int(selected_parameters['min_samples_leaf'])}
            - criterion = squared_error
            - random_state = 42
            """
        )
    st.caption(f"Pipeline verificado: {MODEL_PATH.name}")

st.info(
    "La referencia temporal obtuvo menor MAE que el árbol de decisión. Este resultado es coherente "
    "con el EDA, que mostró una fuerte relación entre el consumo de meses consecutivos. El Decision "
    "Tree funciona como baseline de Machine Learning y servirá como punto de partida para modelos posteriores."
)

