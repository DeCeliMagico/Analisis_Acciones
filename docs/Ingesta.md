# Ingesta de datos (Yahoo Finance)

## Estado: ✅ COMPLETADO (Abril 2026)

**Output**:
- 139 símbolos ingestionados (US equity).
- 1.324M filas de OHLCV históricos.
- Parquet por símbolo en `data/bronze/`.
- Granularidad diaria verificada y estable.
- Próxima fase: [Procesamiento](Procesamiento.md).

---

## Estado actual

La fase de ingesta esta completada para esta etapa inicial:

1. Ingesta individual funcionando.
2. Ingesta masiva funcionando con lista grande de simbolos.
3. Guardado en Parquet dentro del Data Lake local.
4. Granularidad diaria verificada y estable.

## Scripts implementados

1. `scripts/ingesta/ingesta_individual.py`
2. `scripts/ingesta/ingesta_masiva.py`

## Fuente de datos

Endpoint base usado:

```text
https://query1.finance.yahoo.com/v8/finance/chart/{SYMBOL}?interval=1d&period1={period1}&period2={period2}
```

- `interval=1d`: velas diarias.
- `period1/period2`: rango explicito para evitar downsampling (3mo).

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

Volumen actual observado:

- 139 simbolos
- ~1.32M filas totales en Bronze
- ~9.5k filas medias por simbolo

## Como ejecutar

```powershell
python scripts/ingesta/ingesta_individual.py
python scripts/ingesta/ingesta_masiva.py
```

## Siguiente paso

Fase de ingesta cerrada para este MVP.

Estado posterior:

1. Silver de clasificacion ya generado y validado.
2. Proximo paso del proyecto: entrenamiento baseline de clasificacion con split temporal.
