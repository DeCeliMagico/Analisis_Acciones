# Resumen simple de la ingesta (Yahoo)

## Objetivo actual

Hacer una ingesta de una sola accion (AAPL) desde Yahoo Finance y guardarla en el Data Lake local en formato Parquet.

## Lo que ya tenemos

1. Script para una accion: `scripts/ingesta/ingesta_individual.py`.
2. Dependencias en `requirements.txt`.
3. Formato de salida: Parquet.

## Fuente de datos

Endpoint usado:

```text
https://query1.finance.yahoo.com/v8/finance/chart/AAPL?interval=1d&range=max
```

- `interval=1d`: velas diarias.
- `range=max`: historico maximo disponible.

## Columnas que guardamos

1. ts_event_utc
2. open
3. high
4. low
5. close
6. volume
7. symbol

## Flujo de la ingesta

1. Llamar a la API de Yahoo.
2. Leer `timestamp` y `quote` (open/high/low/close/volume).
3. Convertir listas a tabla.
4. Limpiar nulos y ordenar por fecha.
5. Guardar parquet en Bronze.

## Donde se guarda

Salida actual:

- `data/bronze/{symbol}_1d.parquet`

Ejemplo:

- `data/bronze/AAPL_1d.parquet`

## Como ejecutar

```powershell
python scripts/ingesta/ingesta_individual.py
```

## Proximo paso

Pasar de 1 accion a una lista de simbolos (10-20) con el mismo flujo.
