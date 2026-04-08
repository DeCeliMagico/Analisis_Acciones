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

## Documentos de trabajo actuales

- `docs/Ingesta_Individual.md`: resumen simple del estado actual de ingesta.

## Roadmap por fases

1. Fase 0: diseño de contrato de datos y catálogo de símbolos.
2. Fase 1: ingesta histórica masiva (2+ fuentes) y carga Bronze.
3. Fase 2: limpieza + normalización a Silver.
4. Fase 3: dataset Gold y primer dashboard BI (opcional).
5. Fase 4: baseline de regresión y evaluación temporal.

## Criterios de calidad de datos

- Completeness: no huecos inesperados por símbolo/fecha.
- Consistency: OHLC válidos (High >= max(Open, Close), Low <= min(Open, Close)).
- Uniqueness: una fila por símbolo-fecha-fuente (o resolución definida).
- Timeliness: delta de actualización alineado a calendario de mercado.

## Siguientes pasos inmediatos

- Ingesta individual y masiva ya implementadas con Yahoo Finance.
- Datos guardados en `data/bronze` en formato Parquet.
- Siguiente fase: procesamiento Silver (unificacion logica, limpieza y features).
- Preparar dataset para analisis y modelo de regresion (precio o subida/bajada).
