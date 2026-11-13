-- Esquema local PostgreSQL para los resultados oficiales del módulo IA (Hito 3).
-- Es idempotente: no borra ni sustituye datos existentes.

CREATE TABLE IF NOT EXISTS model_runs (
    run_id VARCHAR(64) PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL,
    run_name TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT,
    dataset_name TEXT NOT NULL,
    target_name TEXT NOT NULL,
    selection_metric TEXT NOT NULL,
    validation_mae DOUBLE PRECISION NOT NULL,
    validation_rmse DOUBLE PRECISION NOT NULL,
    validation_r2 DOUBLE PRECISION NOT NULL,
    validation_wape DOUBLE PRECISION NOT NULL,
    test_mae DOUBLE PRECISION NOT NULL,
    test_rmse DOUBLE PRECISION NOT NULL,
    test_r2 DOUBLE PRECISION NOT NULL,
    test_wape DOUBLE PRECISION NOT NULL,
    train_end_period INTEGER NOT NULL,
    validation_start_period INTEGER NOT NULL,
    validation_end_period INTEGER NOT NULL,
    test_start_period INTEGER NOT NULL,
    test_end_period INTEGER NOT NULL,
    random_state INTEGER NOT NULL,
    cv_strategy TEXT NOT NULL,
    hyperparameters JSONB NOT NULL,
    features JSONB NOT NULL,
    manifest JSONB NOT NULL,
    CONSTRAINT model_runs_period_order CHECK (
        train_end_period < validation_start_period
        AND validation_start_period <= validation_end_period
        AND validation_end_period < test_start_period
        AND test_start_period <= test_end_period
    )
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id BIGSERIAL PRIMARY KEY,
    run_id VARCHAR(64) NOT NULL REFERENCES model_runs(run_id) ON DELETE CASCADE,
    distribuidora TEXT NOT NULL,
    ubigeo TEXT NOT NULL,
    tarifa TEXT NOT NULL,
    periodo INTEGER NOT NULL,
    target_periodo INTEGER NOT NULL,
    target_real_mwh DOUBLE PRECISION NOT NULL,
    prediction_mwh DOUBLE PRECISION NOT NULL,
    residual_mwh DOUBLE PRECISION NOT NULL,
    absolute_error_mwh DOUBLE PRECISION NOT NULL CHECK (absolute_error_mwh >= 0),
    percentage_error DOUBLE PRECISION,
    CONSTRAINT predictions_period_order CHECK (periodo < target_periodo),
    CONSTRAINT predictions_unique_series_period UNIQUE (
        run_id, distribuidora, ubigeo, tarifa, periodo, target_periodo
    )
);

CREATE INDEX IF NOT EXISTS idx_predictions_run_id ON predictions(run_id);
CREATE INDEX IF NOT EXISTS idx_predictions_distribuidora ON predictions(distribuidora);
CREATE INDEX IF NOT EXISTS idx_predictions_ubigeo ON predictions(ubigeo);
CREATE INDEX IF NOT EXISTS idx_predictions_tarifa ON predictions(tarifa);
CREATE INDEX IF NOT EXISTS idx_predictions_target_periodo ON predictions(target_periodo);

CREATE TABLE IF NOT EXISTS segment_metrics (
    metric_id BIGSERIAL PRIMARY KEY,
    run_id VARCHAR(64) NOT NULL REFERENCES model_runs(run_id) ON DELETE CASCADE,
    segment_type TEXT NOT NULL,
    segment_value JSONB NOT NULL,
    n_observations INTEGER NOT NULL CHECK (n_observations > 0),
    mae DOUBLE PRECISION NOT NULL CHECK (mae >= 0),
    rmse DOUBLE PRECISION NOT NULL CHECK (rmse >= 0),
    r2 DOUBLE PRECISION,
    wape DOUBLE PRECISION,
    mean_residual DOUBLE PRECISION,
    CONSTRAINT segment_metrics_unique_segment UNIQUE (run_id, segment_type, segment_value)
);

CREATE INDEX IF NOT EXISTS idx_segment_metrics_run_id ON segment_metrics(run_id);
CREATE INDEX IF NOT EXISTS idx_segment_metrics_type ON segment_metrics(segment_type);
