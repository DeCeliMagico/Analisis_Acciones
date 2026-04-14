# Proyecto Big Data de Acciones

Este repositorio inicia un proyecto end-to-end para:

1. Ingestar de forma masiva precios de acciones desde múltiples fuentes públicas.
2. Almacenar datos crudos e históricos en un Data Lake.
3. Limpiar, normalizar y enriquecer los datos.
4. Evaluar si el procesamiento debe ser local o distribuido según volumen y latencia.
5. (Opcional) Exponer analítica en una capa BI.
6. Entrenar un modelo de regresión para predicción/estimación sobre series de mercado.

## Objetivo del MVP

Construir un pipeline reproducible con enfoque Bronze/Silver/Gold:

- Bronze: datos crudos por fuente (sin transformar).
- Silver: datos limpios, tipados, deduplicados y alineados por calendario.
- Gold: datasets analíticos y de entrenamiento para BI/ML.

## Alcance funcional inicial

- Ingesta masiva histórica de OHLCV (Open, High, Low, Close, Volume).
- Cobertura de múltiples símbolos (US al inicio, escalable a otros mercados).
- Escritura en Data Lake por particiones (fecha, mercado, símbolo).
- Trazabilidad total: fuente, timestamp de ingesta, versión de esquema.

## Arquitectura propuesta (iterativa)

### 1) Ingesta

- Conectores HTTP para APIs públicas.
- Descargas batch por ventanas temporales (backfill + incremental diario).
- Persistencia inmediata en zona Bronze.

### 2) Data Lake

Tecnologías candidatas (a decidir según coste/operativa):

- Local para prototipo: MinIO (S3-compatible) + Parquet.
- Cloud: S3 / ADLS / GCS + formato Parquet.
- Formato de tabla opcional para ACID/versionado: Delta Lake o Apache Iceberg.

### 3) Procesamiento y calidad

Estrategia de decisión local vs distribuido:

- Local (Pandas/Polars) si:
  - volumen < ~50-100 GB histórico,
  - ventana de proceso diaria corta,
  - baja concurrencia.
- Distribuido (Spark/Dask/Flink) si:
  - volumen crece por encima de ese rango,
  - SLAs estrictos,
  - necesidad de joins pesados y re-procesos frecuentes.

### 4) Consumo analítico

- Datasets Gold para KPI de mercado, retornos, volatilidad, drawdown, rolling stats.
- Exposición para BI (Power BI, Superset, Metabase).

### 5) ML (Regresión)

- Baseline: regresión lineal/regularizada sobre features temporales.
- Iteraciones: modelos de árbol boosting (XGBoost/LightGBM) y validación walk-forward.
- Métricas: RMSE, MAE, MAPE, estabilidad por régimen de mercado.

## Estructura sugerida del proyecto

```text
.
├─ README.md
├─ docs/
│  └─ ingesta-masiva-fuentes.md
├─ data/
│  ├─ bronze/
│  ├─ silver/
│  └─ gold/
├─ pipelines/
│  ├─ ingest/
│  ├─ transform/
│  └─ features/
├─ notebooks/
├─ models/
└─ tests/
```

## Status actual (Abril 2026)

✅ **Completado**:
- Bronze: 139 símbolos, 1.324M filas, granularidad diaria.
- Silver: 1.321M filas procesadas, 12 features + target_updown_t1 (clasificación binaria).
  
🟡 **En curso**:
- Entrenamiento de clasificador binario con AutoGluon baseline.
  - Split temporal: 70% train / 15% valid / 15% test.
  - Tiempo estimado: ~1 hora (ajustable con time_limit).
  - Salida: modelo en `modelos/Market_AI_*`, métricas en `evaluaciones/metricas_clasificacion.json`.

⏳ **Próximas fases**:
- Evaluación de rendimiento (AUC, precision, recall, F1).
- Regresión de retornos (target_ret_t1).
- Ensemble clasificación + regresión para ranking final.

## Documentos de trabajo actuales

- `docs/Ingesta.md`: resumen de ingesta y estado.
- `docs/Procesamiento.md`: feature engineering y limpieza.
- `docs/clasificacion_binaria.md`: AutoGluon, métricas, interpretación.
- `docs/Entrenamiento.md`: instrucciones de ejecución del pipeline.
- `docs/Objetivos.md`: mapa de fases y resultados esperados.

## Roadmap por fases

1. ✅ Fase 0-2: ingesta Bronze + procesamiento Silver.
2. 🟡 Fase 3: entrenamiento clasificación + evaluación.
3. ⏳ Fase 4: modelo de regresión y optimización.
4. ⏳ Fase 5: ensemble final y producción.

## Criterios de calidad de datos

- Completeness: no huecos inesperados por símbolo/fecha.
- Consistency: OHLC válidos (High >= max(Open, Close), Low <= min(Open, Close)).
- Uniqueness: una fila por símbolo-fecha-fuente (o resolución definida).
- Timeliness: delta de actualización alineado a calendario de mercado.

## Siguientes pasos inmediatos

- Ingesta individual y masiva ya implementadas con Yahoo Finance.
- Datos guardados en `data/bronze` en formato Parquet.
- Ingesta diaria corregida con `period1/period2` (evitando downsampling de Yahoo).
- Procesamiento Silver de clasificacion completado (features + `target_updown_t1`).
- Dataset generado en `data/silver` con nombre versionado por fecha y hora.
- Estado actual: listo para comenzar entrenamiento baseline con split temporal.
