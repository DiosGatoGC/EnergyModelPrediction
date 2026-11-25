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
PREDICTIONS_BASELINE_PATH = OUTPUT_DIR / "predicciones_test.parquet"
METRICS_BASELINE_PATH = OUTPUT_DIR / "metricas_baseline.parquet"
DIST_METRICS_BASELINE_PATH = OUTPUT_DIR / "metricas_por_distribuidora.parquet"
TARIFF_METRICS_BASELINE_PATH = OUTPUT_DIR / "metricas_por_tarifa.parquet"

# Rutas Hito 3
PREDICTIONS_H3_PATH = OUTPUT_DIR / "predicciones_test_comparativa.parquet"
HIST_H3_PATH = OUTPUT_DIR / "serie_historica_real_completa.parquet"
METRICS_H3_PATH = OUTPUT_DIR / "metricas_hito3_comparativa.parquet"
METRICS_VAL_H3_PATH = OUTPUT_DIR / "metricas_hito3_validation.parquet"
DIST_METRICS_H3_PATH = OUTPUT_DIR / "metricas_hito3_por_distribuidora.parquet"
TARIFF_METRICS_H3_PATH = OUTPUT_DIR / "metricas_hito3_por_tarifa.parquet"
MANIFEST_H3_PATH = OUTPUT_DIR / "hito3_run_manifest.json"

@st.cache_data
def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)

@st.cache_data
def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)

@st.cache_data
def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    for col in ("DISTRIBUIDORA", "UBIGEO", "TARIFA"):
        if col in df.columns:
            df[col] = df[col].astype("string")
    return df


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
    "Selecciona el Modo / Fase:",
    [
        "Producto: Reconstrucción y Pronóstico Futuro",
        "Hito 3: Modelos Avanzados (campeón por VALIDATION)",
        "Hito 2: Baseline Inicial (Decision Tree vs Referencia)",
    ],
    index=0
)

CUBE_PATH = ROOT / "reports" / "eda_outputs" / "resumen_distribuidora_ubigeo_tarifa_periodo.parquet"
CHAMPION_MODEL_PATH = MODELS_DIR / "modelo_campeon_hito3.joblib"
GEOJSON_PATH = ROOT / "metadata" / "geo" / "distritos_concesion.geojson"


@st.cache_data
def load_geojson_concesion() -> tuple[dict, pd.DataFrame]:
    with GEOJSON_PATH.open(encoding="utf-8") as f:
        geo = json.load(f)
    meta_rows = []
    for feat in geo.get("features", []):
        p = feat.get("properties", {})
        u = str(p.get("ubigeo", "")).strip().zfill(6)
        p["ubigeo"] = u
        meta_rows.append({
            "UBIGEO": u,
            "DISTRITO": p.get("nombre") or p.get("nombdist") or u,
            "PROVINCIA": p.get("nombre_provincia") or "",
            "DEPARTAMENTO": p.get("nombre_departamento") or "",
        })
    return geo, pd.DataFrame(meta_rows).drop_duplicates("UBIGEO")


def add_months_ym(ym: int, delta: int) -> int:
    y, m = divmod(int(ym), 100)
    idx = y * 12 + (m - 1) + delta
    ny, nm = divmod(idx, 12)
    return ny * 100 + (nm + 1)


@st.cache_resource
def load_champion_model():
    import joblib
    return joblib.load(CHAMPION_MODEL_PATH)


@st.cache_data(show_spinner="Detectando anomalías, reconstruyendo serie y calculando pronóstico futuro...")
def build_product_series(
    horizon_months: int,
    gwh_ratio_threshold: float = 0.80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cube = pd.read_parquet(CUBE_PATH)
    cube["PERIODO"] = cube["PERIODO"].astype(int)
    model = load_champion_model()
    feature_names = list(model.feature_name_)

    # Detectar meses anómalos por caída brusca de demanda GWh vs mediana móvil centrada de 5 meses
    monthly_tot = (
        cube.groupby(["DISTRIBUIDORA", "PERIODO"], as_index=False)["consumo_total_mwh"]
        .sum()
        .sort_values(["DISTRIBUIDORA", "PERIODO"])
    )
    anomalous_periods: dict[str, set[int]] = {"ENOSA": set(), "HIDRANDINA": set()}
    anomaly_summary_rows = []
    for dist in monthly_tot["DISTRIBUIDORA"].unique():
        sub = monthly_tot.loc[monthly_tot["DISTRIBUIDORA"] == dist].sort_values("PERIODO").copy()
        sub["roll_med_gwh"] = sub["consumo_total_mwh"].rolling(5, center=True, min_periods=2).median()
        sub["gwh_ratio"] = sub["consumo_total_mwh"] / sub["roll_med_gwh"]
        bad = sub.loc[sub["gwh_ratio"] < gwh_ratio_threshold]
        anomalous_periods[str(dist)] = set(bad["PERIODO"].astype(int).tolist())
        for r in bad.itertuples(index=False):
            anomaly_summary_rows.append({
                "DISTRIBUIDORA": str(r.DISTRIBUIDORA),
                "PERIODO": int(r.PERIODO),
                "Real_Original_GWh": float(r.consumo_total_mwh) / 1e3,
                "Mediana_Movil_GWh": float(r.roll_med_gwh) / 1e3,
                "Ratio_GWh": float(r.gwh_ratio),
            })

    # Lookup limpio para construir lags sin contaminar con meses anómalos
    lookup: dict[tuple[tuple[str, str, str], int], tuple[float, float, float]] = {}
    series_by_dist: dict[str, set[tuple[str, str, str]]] = {"ENOSA": set(), "HIDRANDINA": set()}
    last_known_meta: dict[tuple[str, str, str], tuple[float, float]] = {}

    for row in cube.itertuples(index=False):
        dist = str(row.DISTRIBUIDORA)
        key = (dist, str(row.UBIGEO), str(row.TARIFA))
        per = int(row.PERIODO)
        series_by_dist.setdefault(dist, set()).add(key)
        last_known_meta[key] = (float(row.n_registros), float(row.consumo_mediano))
        if per not in anomalous_periods.get(dist, set()):
            lookup[(key, per)] = (float(row.consumo_total_mwh), float(row.n_registros), float(row.consumo_mediano))

    def predict_step(keys: list[tuple[str, str, str]], base_period: int, target_period: int, tipo: str) -> list[dict]:
        rows, valid_keys, base_consumos = [], [], []
        for key in keys:
            base_val = lookup.get((key, base_period))
            if base_val is None:
                continue
            c_t, n_reg, c_med = base_val
            l1 = lookup.get((key, add_months_ym(base_period, -1)))
            l2 = lookup.get((key, add_months_ym(base_period, -2)))
            l3 = lookup.get((key, add_months_ym(base_period, -3)))
            l12 = lookup.get((key, add_months_ym(base_period, -12)))
            lag1 = l1[0] if l1 is not None else np.nan
            lag2 = l2[0] if l2 is not None else np.nan
            lag3 = l3[0] if l3 is not None else np.nan
            lag12 = l12[0] if l12 is not None else np.nan
            mes = base_period % 100
            valid_vals = [v for v in (c_t, lag1, lag2) if not np.isnan(v)]
            rows.append({
                "DISTRIBUIDORA": key[0],
                "UBIGEO": key[1],
                "TARIFA": key[2],
                "MES": mes,
                "MES_SIN": float(np.sin(2 * np.pi * mes / 12.0)),
                "MES_COS": float(np.cos(2 * np.pi * mes / 12.0)),
                "n_registros": n_reg,
                "consumo_mediano": c_med,
                "consumo_total_mwh": c_t,
                "lag_1": lag1,
                "lag_2": lag2,
                "lag_3": lag3,
                "lag_12": lag12,
                "RATIO_LAG1": (c_t + 1.0) / (lag1 + 1.0) if not np.isnan(lag1) else np.nan,
                "RATIO_LAG2": (c_t + 1.0) / (lag2 + 1.0) if not np.isnan(lag2) else np.nan,
                "RATIO_LAG3": (c_t + 1.0) / (lag3 + 1.0) if not np.isnan(lag3) else np.nan,
                "ROLLING_MEAN_3": float(np.mean(valid_vals)),
                "HAS_LAG1": int(not np.isnan(lag1)),
                "HAS_LAG2": int(not np.isnan(lag2)),
                "HAS_LAG3": int(not np.isnan(lag3)),
                "HAS_LAG12": int(not np.isnan(lag12)),
            })
            valid_keys.append((key, n_reg, c_med))
            base_consumos.append(c_t)

        if not rows:
            return []
        X_step = pd.DataFrame(rows)
        for col in ("DISTRIBUIDORA", "UBIGEO", "TARIFA"):
            X_step[col] = X_step[col].astype("category")
        pred_delta = model.predict(X_step[feature_names])
        pred_mwh = np.clip(np.expm1(np.log1p(np.array(base_consumos)) + pred_delta), 0, None)

        out = []
        for (key, n_reg, c_med), val in zip(valid_keys, pred_mwh):
            val_f = float(val)
            lookup[(key, target_period)] = (val_f, n_reg, c_med)
            out.append({
                "DISTRIBUIDORA": key[0],
                "UBIGEO": key[1],
                "TARIFA": key[2],
                "PERIODO": target_period,
                "n_registros": int(n_reg),
                "consumo_total_mwh": val_f,
                "TIPO": tipo,
            })
        return out

    generated: list[dict] = []

    # 1. Reconstruir cronológicamente anomalías detectadas y huecos de Hidrandina hasta 2026-08
    hid_keys_sorted = sorted(series_by_dist.get("HIDRANDINA", set()))
    # Recorrer desde 202211 hasta 202608 en orden cronológico
    curr = 202210
    while curr < 202608:
        nxt = add_months_ym(curr, 1)
        is_missing_hid = nxt == 202211 or nxt >= 202508
        is_anom_hid = nxt in anomalous_periods.get("HIDRANDINA", set())
        if is_missing_hid or is_anom_hid:
            tipo_lbl = "Reconstruido (Anomalía)" if is_anom_hid else "Reconstruido (Faltante)"
            generated.extend(predict_step(hid_keys_sorted, curr, nxt, tipo_lbl))
        curr = nxt

    # Reconstruir anomalías si las hubiera en ENOSA
    enosa_keys_sorted = sorted(series_by_dist.get("ENOSA", set()))
    for anom_p in sorted(anomalous_periods.get("ENOSA", set())):
        prev_p = add_months_ym(anom_p, -1)
        generated.extend(predict_step(enosa_keys_sorted, prev_p, anom_p, "Reconstruido (Anomalía)"))

    # 2. Pronóstico futuro (2026-09 en adelante) para ambas distribuidoras sobre la serie limpia
    all_active_keys = enosa_keys_sorted + hid_keys_sorted
    curr = 202608
    for _ in range(horizon_months):
        nxt = add_months_ym(curr, 1)
        generated.extend(predict_step(all_active_keys, curr, nxt, "Pronóstico Futuro"))
        curr = nxt

    # Conservar original marcado (Real vs Anómalo Original) para comparación visual
    base_df = cube[["DISTRIBUIDORA", "UBIGEO", "TARIFA", "PERIODO", "n_registros", "consumo_total_mwh"]].copy()
    base_df["TIPO"] = np.where(
        base_df.apply(lambda r: int(r["PERIODO"]) in anomalous_periods.get(str(r["DISTRIBUIDORA"]), set()), axis=1),
        "Real (Anomalía Original)",
        "Real",
    )
    gen_df = pd.DataFrame(generated)
    combined = pd.concat([base_df, gen_df], ignore_index=True) if not gen_df.empty else base_df
    anom_df = pd.DataFrame(anomaly_summary_rows)
    return combined.sort_values(["DISTRIBUIDORA", "UBIGEO", "TARIFA", "PERIODO"]).reset_index(drop=True), anom_df


@st.cache_data
def load_ubigeo_labels() -> dict[str, str]:
    labels: dict[str, str] = {}
    if GEOJSON_PATH.is_file():
        _, geo_meta = load_geojson_concesion()
        for r in geo_meta.itertuples(index=False):
            u = str(r.UBIGEO).strip().zfill(6)
            dist_n = str(r.DISTRITO).strip().title()
            dep_n = str(r.DEPARTAMENTO).strip().title()
            labels[u] = f"{dist_n} ({dep_n}) — {u}" if dep_n else f"{dist_n} — {u}"
    return labels


is_product = hito_selection.startswith("Producto")

# Carga condicional según selección.
if is_product:
    horizon_months = st.sidebar.slider("Horizonte futuro (meses tras 2026-08)", min_value=1, max_value=6, value=4)
    gwh_drop_threshold = st.sidebar.slider(
        "Umbral caída de GWh (ratio vs mediana 5m)",
        min_value=0.50, max_value=0.95, value=0.80, step=0.05,
    )
    predictions, anomaly_df = build_product_series(horizon_months, gwh_drop_threshold)
    history_df = None
    metrics_test = pd.DataFrame()
    metrics_val = None
    dist_metrics = pd.DataFrame()
    tariff_metrics = pd.DataFrame()
    ubigeo_metrics = pd.DataFrame()
    h3_manifest = {}
    data_source_label = "Cubo + Modelo Hito 3"
    is_hito3 = False
elif "Hito 3" in hito_selection:
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
        metrics_test = load_parquet(METRICS_H3_PATH)
        metrics_val = load_parquet(METRICS_VAL_H3_PATH) if METRICS_VAL_H3_PATH.is_file() else None
        dist_metrics = load_parquet(DIST_METRICS_H3_PATH)
        tariff_metrics = load_parquet(TARIFF_METRICS_H3_PATH)
        ubigeo_metrics = load_parquet(OUTPUT_DIR / "hito3_errores_por_ubigeo.parquet")
        h3_manifest = load_json(MANIFEST_H3_PATH) if MANIFEST_H3_PATH.is_file() else {}
        h3_manifest["_segmentos_por_periodo"] = load_parquet(OUTPUT_DIR / "hito3_errores_por_mes.parquet").to_dict(orient="records")
        h3_manifest["_segmentos_por_rango"] = load_parquet(OUTPUT_DIR / "hito3_errores_por_rango_consumo.parquet").to_dict(orient="records")
        data_source_label = "archivos locales"
    is_hito3 = True
else:
    predictions = load_predictions(PREDICTIONS_BASELINE_PATH)
    history_df = None
    metrics_test = load_parquet(METRICS_BASELINE_PATH)
    metrics_val = None
    dist_metrics = load_parquet(DIST_METRICS_BASELINE_PATH)
    tariff_metrics = load_parquet(TARIFF_METRICS_BASELINE_PATH)
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

ubigeo_labels = load_ubigeo_labels()
raw_ubigeos = filter_base["UBIGEO"].dropna().astype(str).str.strip().str.zfill(6).unique().tolist()
if is_product and ubigeo_labels:
    raw_ubigeos = [u for u in raw_ubigeos if u in ubigeo_labels]
ubigeo_options = ["Todos"] + sorted(raw_ubigeos, key=lambda u: ubigeo_labels.get(u, u))
selected_ubigeo = st.sidebar.selectbox(
    "Distrito (UBIGEO)",
    ubigeo_options,
    format_func=lambda u: "Todos" if u == "Todos" else ubigeo_labels.get(u, u),
)
if selected_ubigeo != "Todos":
    filter_base = filter_base.loc[filter_base["UBIGEO"].astype(str).str.strip().str.zfill(6) == selected_ubigeo]

tariff_options = ["Todas"] + sorted(filter_base["TARIFA"].dropna().unique().tolist())
selected_tariff = st.sidebar.selectbox("Tarifa Eléctrica", tariff_options)
if selected_tariff != "Todas":
    filter_base = filter_base.loc[filter_base["TARIFA"] == selected_tariff]

filtered = filter_base
st.sidebar.metric("Observaciones seleccionadas", f"{len(filtered):,}")

if filtered.empty:
    st.warning("No existen registros para la combinación de filtros seleccionada.")
    st.stop()

# Vista Producto
if is_product:
    st.markdown("### Reconstrucción de Anomalías y Pronóstico Futuro")

    real_mwh = filtered.loc[filtered["TIPO"].isin(["Real", "Real (Anomalía Original)"]), "consumo_total_mwh"].sum()
    rec_anom_mwh = filtered.loc[filtered["TIPO"] == "Reconstruido (Anomalía)", "consumo_total_mwh"].sum()
    rec_miss_mwh = filtered.loc[filtered["TIPO"] == "Reconstruido (Faltante)", "consumo_total_mwh"].sum()
    fut_mwh = filtered.loc[filtered["TIPO"] == "Pronóstico Futuro", "consumo_total_mwh"].sum()

    pk = st.columns(4)
    pk[0].metric("Consumo Real Reportado", f"{real_mwh / 1e3:,.2f} GWh")
    pk[1].metric("Anomalías Reconstruidas", f"{rec_anom_mwh / 1e3:,.2f} GWh")
    pk[2].metric("Meses Faltantes Reconstruidos", f"{rec_miss_mwh / 1e3:,.2f} GWh")
    pk[3].metric(f"Pronóstico Futuro (+{horizon_months}m)", f"{fut_mwh / 1e3:,.2f} GWh")

    ts_prod = (
        filtered.groupby(["DISTRIBUIDORA", "PERIODO", "TIPO"], as_index=False)["consumo_total_mwh"]
        .sum()
        .sort_values(["DISTRIBUIDORA", "PERIODO"])
    )
    ts_prod["Periodo"] = format_period(ts_prod["PERIODO"])
    ts_prod["Energia_GWh"] = ts_prod["consumo_total_mwh"] / 1e3

    fig_prod = go.Figure()
    color_map = {"ENOSA": "#0284c7", "HIDRANDINA": "#16a34a"}

    for dist in sorted(ts_prod["DISTRIBUIDORA"].unique()):
        sub_d = ts_prod.loc[ts_prod["DISTRIBUIDORA"] == dist]
        c_hex = color_map.get(dist, "#334155")

        # 1. Serie Real Observada (línea sólida)
        orig_d = sub_d.loc[sub_d["TIPO"].isin(["Real", "Real (Anomalía Original)"])].sort_values("PERIODO")
        anom_pts = sub_d.loc[sub_d["TIPO"] == "Real (Anomalía Original)"].sort_values("PERIODO")
        if not orig_d.empty:
            fig_prod.add_trace(go.Scatter(
                x=orig_d["Periodo"], y=orig_d["Energia_GWh"],
                mode="lines+markers", name=f"{dist} — Real (Observado)",
                line={"color": c_hex, "width": 2.6, "dash": "solid"},
                marker={"size": 5},
            ))
        if not anom_pts.empty:
            fig_prod.add_trace(go.Scatter(
                x=anom_pts["Periodo"], y=anom_pts["Energia_GWh"],
                mode="markers", name=f"{dist} — Anomalía detectada",
                marker={"size": 9, "symbol": "x", "color": "#ef4444"},
            ))

        # 2. Serie Reconstruida por el Modelo (línea punteada SOLO en los huecos y anomalías)
        clean_d = sub_d.loc[sub_d["TIPO"].isin(["Real", "Reconstruido (Anomalía)", "Reconstruido (Faltante)"])].sort_values("PERIODO").reset_index(drop=True)
        rec_only = sub_d.loc[sub_d["TIPO"].isin(["Reconstruido (Anomalía)", "Reconstruido (Faltante)"])].sort_values("PERIODO")
        if not rec_only.empty and not clean_d.empty:
            is_rec = (clean_d["TIPO"] != "Real").to_numpy()
            show_seg = is_rec.copy()
            show_seg[:-1] |= is_rec[1:]
            show_seg[1:] |= is_rec[:-1]
            rec_y = [val if keep else None for val, keep in zip(clean_d["Energia_GWh"], show_seg)]
            fig_prod.add_trace(go.Scatter(
                x=clean_d["Periodo"], y=rec_y,
                mode="lines", name=f"{dist} — Reconstrucción (Modelo)",
                line={"color": "#f59e0b", "width": 2.4, "dash": "dot"},
                connectgaps=False,
            ))
            fig_prod.add_trace(go.Scatter(
                x=rec_only["Periodo"], y=rec_only["Energia_GWh"],
                mode="markers", name=f"{dist} — Puntos Reconstruidos",
                marker={"size": 8, "symbol": "diamond", "color": "#f59e0b"},
            ))

        # 3. Pronóstico Futuro (2026-09+)
        fut_d = sub_d.loc[sub_d["TIPO"] == "Pronóstico Futuro"].sort_values("PERIODO")
        if not fut_d.empty:
            anchor = clean_d.loc[clean_d["PERIODO"] == 202608]
            fut_plot = pd.concat([anchor, fut_d], ignore_index=True) if not anchor.empty else fut_d
            fig_prod.add_trace(go.Scatter(
                x=fut_plot["Periodo"], y=fut_plot["Energia_GWh"],
                mode="lines+markers", name=f"{dist} — Pronóstico Futuro",
                line={"color": c_hex, "width": 2.8, "dash": "dash"},
                marker={"size": 6, "symbol": "star"},
            ))

    fig_prod.update_layout(
        title="Serie Temporal",
        xaxis_title="Periodo Mensual",
        yaxis_title="Energía Mensual (GWh)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    st.plotly_chart(fig_prod, width="stretch")

    if not anomaly_df.empty:
        st.markdown("#### Anomalías Históricas Detectadas y Reconstrucción Contrafactual")
        rec_anom_totals = (
            ts_prod.loc[ts_prod["TIPO"] == "Reconstruido (Anomalía)", ["DISTRIBUIDORA", "PERIODO", "Energia_GWh"]]
            .rename(columns={"Energia_GWh": "Reconstruido_GWh"})
        )
        anom_view = anomaly_df.merge(rec_anom_totals, on=["DISTRIBUIDORA", "PERIODO"], how="left")
        anom_view["Diferencia_GWh"] = anom_view["Reconstruido_GWh"] - anom_view["Real_Original_GWh"]
        st.dataframe(anom_view.round(3), width="stretch", hide_index=True)

    # Mapa territorial (GeoJSON)
    if GEOJSON_PATH.is_file():
        st.markdown("### Mapa Territorial (GeoJSON)")
        geo_data, geo_meta = load_geojson_concesion()
        map_metric = st.radio(
            "Métrica a visualizar en el mapa:",
            ["Demanda Total Limpia (GWh)", "Pronóstico Futuro (GWh)", "Último Mes Real vs Reconstruido (GWh)"],
            horizontal=True,
        )

        clean_mask = filtered["TIPO"].isin(["Real", "Reconstruido (Anomalía)", "Reconstruido (Faltante)"])
        fut_mask = filtered["TIPO"] == "Pronóstico Futuro"
        last_hist_mask = clean_mask & (filtered["PERIODO"] == 202608)

        if map_metric == "Pronóstico Futuro (GWh)":
            map_src = filtered.loc[fut_mask]
        elif map_metric == "Último Mes Real vs Reconstruido (GWh)":
            map_src = filtered.loc[last_hist_mask]
        else:
            map_src = filtered.loc[clean_mask]

        agg_geo = (
            map_src.groupby("UBIGEO", as_index=False)
            .agg(
                consumo_mwh=("consumo_total_mwh", "sum"),
                DISTRIBUIDORA=("DISTRIBUIDORA", "first"),
            )
        )
        agg_geo["UBIGEO"] = agg_geo["UBIGEO"].astype(str).str.strip().str.zfill(6)
        agg_geo["Energia_GWh"] = agg_geo["consumo_mwh"] / 1e3

        map_df = geo_meta.merge(agg_geo, on="UBIGEO", how="inner")
        map_df["Energia_GWh"] = map_df["Energia_GWh"].fillna(0.0)
        map_df["Color_Log"] = np.log1p(map_df["Energia_GWh"])

        fig_map = go.Figure()
        if not map_df.empty:
            fig_map.add_trace(go.Choropleth(
                geojson=geo_data,
                locations=map_df["UBIGEO"],
                z=map_df["Color_Log"],
                featureidkey="properties.ubigeo",
                colorscale="YlOrRd",
                marker_line_color="#334155",
                marker_line_width=0.6,
                colorbar=dict(
                    title="log1p(GWh)",
                    thickness=14,
                    len=0.75,
                    x=0.92,
                ),
                customdata=np.stack([
                    map_df["DISTRITO"],
                    map_df["PROVINCIA"],
                    map_df["DEPARTAMENTO"],
                    map_df["DISTRIBUIDORA"],
                    map_df["Energia_GWh"],
                ], axis=-1),
                hovertemplate=(
                    "<b>%{customdata[0]}</b> (%{location})<br>"
                    "Provincia: %{customdata[1]} | Dpto: %{customdata[2]}<br>"
                    "Distribuidora: %{customdata[3]}<br>"
                    "Energía: <b>%{customdata[4]:.3f} GWh</b><extra></extra>"
                ),
            ))
        if selected_ubigeo != "Todos" and not map_df.empty:
            fig_map.update_geos(
                visible=True,
                resolution=50,
                scope="south america",
                showcountries=True,
                countrycolor="#64748b",
                countrywidth=1.2,
                showcoastlines=True,
                coastlinecolor="#64748b",
                coastlinewidth=1.0,
                showland=True,
                landcolor="#0f172a",
                showocean=True,
                oceancolor="#020617",
                showlakes=False,
                fitbounds="locations",
                bgcolor="rgba(0,0,0,0)",
            )
        else:
            fig_map.update_geos(
                visible=True,
                resolution=50,
                scope="south america",
                showcountries=True,
                countrycolor="#64748b",
                countrywidth=1.2,
                showcoastlines=True,
                coastlinecolor="#64748b",
                coastlinewidth=1.0,
                showland=True,
                landcolor="#0f172a",
                showocean=True,
                oceancolor="#020617",
                showlakes=False,
                lataxis_range=[-11.8, -2.5],
                lonaxis_range=[-83.5, -71.5],
                bgcolor="rgba(0,0,0,0)",
            )
        fig_map.update_layout(
            height=820,
            margin=dict(l=0, r=0, t=40, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            title=f"{map_metric} — Distritos Seleccionados ({len(map_df)} de {len(geo_meta)})",
        )
        st.plotly_chart(fig_map, width="stretch")

    st.markdown("#### Resumen Mensual (Real vs Reconstruido vs Pronóstico Futuro)")
    pivot_prod = (
        ts_prod.pivot_table(index="Periodo", columns=["DISTRIBUIDORA", "TIPO"], values="Energia_GWh", aggfunc="sum")
        .sort_index()
    )
    st.dataframe(pivot_prod.round(3), width="stretch")

# Vista Hito 3
elif is_hito3:
    st.markdown("### Evaluación en TEST (2025-04 a 2025-07)")
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
    kpis[1].metric(
        f"MAE {champion_family} (Campeón)",
        f"{champ_row['MAE_MWh']:.3f} MWh",
        delta=f"{-champ_row['MEJORA_MAE_pct']:+.2f}% vs Ref",
        delta_color="inverse",
    )
    kpis[2].metric(
        "MAE LightGBM",
        f"{lgb_row['MAE_MWh']:.3f} MWh",
        delta=f"{-lgb_row['MEJORA_MAE_pct']:+.2f}% vs Ref",
        delta_color="inverse",
    )
    kpis[3].metric(
        "MAE CatBoost",
        f"{cb_row['MAE_MWh']:.3f} MWh",
        delta=f"{-cb_row['MEJORA_MAE_pct']:+.2f}% vs Ref",
        delta_color="inverse",
    )
    kpis[4].metric(
        f"WAPE {champion_family}",
        f"{champ_row['WAPE_pct']:.2f}%",
        delta=f"{-champ_row['MEJORA_MAE_pct']:+.2f}% MAE vs Ref",
        delta_color="inverse",
    )
    st.caption(f"Campeón: {champion_name} (seleccionado en VALIDATION).")
    
    st.markdown("#### Comparativa de Modelos")
    tab1, tab2 = st.tabs([
        "TEST",
        "VALIDATION",
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
            title="Distribución de residuos (recortada a ±100 MWh)"
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

# Vista Hito 2
else:
    st.markdown("### Baseline — Decision Tree (Hito 2)")
    test_metrics = metrics_test.loc[metrics_test["CONJUNTO"].astype(str).str.upper() == "TEST"]
    ref_row = test_metrics.loc[test_metrics["MODELO"] == "Referencia temporal"].iloc[0]
    tree_row = test_metrics.loc[test_metrics["MODELO"] == "Decision Tree"].iloc[0]
    
    kpis = st.columns(4)
    kpis[0].metric("MAE Referencia temporal", f"{ref_row['MAE']:.3f} MWh")
    kpis[1].metric(
        "MAE Decision Tree",
        f"{tree_row['MAE']:.3f} MWh",
        delta=f"{-tree_row['MEJORA_MAE_PCT']:+.1f}% vs Ref",
        delta_color="inverse",
    )
    kpis[2].metric("RMSE Decision Tree", f"{tree_row['RMSE']:.3f} MWh")
    kpis[3].metric("R² Decision Tree", f"{tree_row['R2']:.4f}")
    
    st.warning(
        f"En el Hito 2, el DecisionTreeRegressor no superó la referencia temporal "
        f"({tree_row['MEJORA_MAE_PCT']:+.2f}% en MAE), debido a la alta cardinalidad de UBIGEO y la aproximación escalonada sobre niveles absolutos."
    )
    st.dataframe(test_metrics, width="stretch", hide_index=True)
