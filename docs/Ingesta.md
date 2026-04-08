# Resumen simple de la ingesta (Yahoo)

## Estado actual

La fase de ingesta esta completada para esta etapa inicial:

1. Ingesta individual funcionando.
2. Ingesta masiva funcionando con lista grande de simbolos.
3. Guardado en Parquet dentro del Data Lake local.

## Scripts implementados

1. `scripts/ingesta/ingesta_individual.py`
2. `scripts/ingesta/ingesta_masiva.py`

## Fuente de datos

Endpoint base usado:

```text
https://query1.finance.yahoo.com/v8/finance/chart/{SYMBOL}?interval=1d&range=max
```

- `interval=1d`: velas diarias.
- `range=max`: historico maximo disponible.

## Datos guardados

Columnas principales:

1. ts_event_utc
2. open
3. high
4. low
5. close
6. volume
7. symbol

Ruta de salida:

- `data/bronze/{symbol}_1d.parquet`

## Como ejecutar

```powershell
python scripts/ingesta/ingesta_individual.py
python scripts/ingesta/ingesta_masiva.py
```

## Siguiente paso

Pasar a Silver:

1. Unificar lectura de todos los parquet de Bronze.
2. Limpieza de calidad.
3. Crear features para analisis y modelo.
