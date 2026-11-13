-- Consultas de demostración para PostgreSQL local. Seleccionan la corrida más reciente.

-- 1. Última corrida y métricas oficiales del campeón.
SELECT run_id, created_at, model_name, validation_mae, test_mae, test_rmse, test_r2, test_wape
FROM model_runs
ORDER BY created_at DESC
LIMIT 1;

-- 2. Conteo de predicciones TEST de la última corrida.
SELECT COUNT(*) AS predicciones_test
FROM predictions
WHERE run_id = (SELECT run_id FROM model_runs ORDER BY created_at DESC LIMIT 1);

-- 3. MAE promedio por distribuidora, calculado desde predicciones almacenadas.
SELECT distribuidora, COUNT(*) AS observaciones, AVG(absolute_error_mwh) AS mae_mwh
FROM predictions
WHERE run_id = (SELECT run_id FROM model_runs ORDER BY created_at DESC LIMIT 1)
GROUP BY distribuidora
ORDER BY mae_mwh DESC;

-- 4. Errores por tarifa.
SELECT tarifa, COUNT(*) AS observaciones, AVG(absolute_error_mwh) AS mae_mwh
FROM predictions
WHERE run_id = (SELECT run_id FROM model_runs ORDER BY created_at DESC LIMIT 1)
GROUP BY tarifa
ORDER BY mae_mwh DESC;

-- 5. Top 10 peores predicciones.
SELECT distribuidora, ubigeo, tarifa, target_periodo, target_real_mwh, prediction_mwh, absolute_error_mwh
FROM predictions
WHERE run_id = (SELECT run_id FROM model_runs ORDER BY created_at DESC LIMIT 1)
ORDER BY absolute_error_mwh DESC
LIMIT 10;

-- 6. Predicciones de un UBIGEO: sustituya '130101' por el UBIGEO deseado.
SELECT distribuidora, ubigeo, tarifa, target_periodo, target_real_mwh, prediction_mwh, residual_mwh
FROM predictions
WHERE run_id = (SELECT run_id FROM model_runs ORDER BY created_at DESC LIMIT 1)
  AND ubigeo = '130101'
ORDER BY target_periodo, tarifa;

-- 7. Predicciones por TARGET_PERIODO.
SELECT target_periodo, COUNT(*) AS observaciones, AVG(absolute_error_mwh) AS mae_mwh
FROM predictions
WHERE run_id = (SELECT run_id FROM model_runs ORDER BY created_at DESC LIMIT 1)
GROUP BY target_periodo
ORDER BY target_periodo;

-- 8. Métricas segmentadas oficialmente persistidas.
SELECT segment_type, segment_value, n_observations, mae, rmse, r2, wape, mean_residual
FROM segment_metrics
WHERE run_id = (SELECT run_id FROM model_runs ORDER BY created_at DESC LIMIT 1)
ORDER BY segment_type, mae DESC;
