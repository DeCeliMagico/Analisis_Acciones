# Proyecto de Predicción de Acciones

Proyecto para probar modelos de machine learning sobre acciones usando datos diarios de Yahoo Finance. El objetivo actual es generar señales de compra ficticias a partir de modelos de clasificación binaria.

El enfoque que mejor ha funcionado hasta ahora es entrenar **un modelo por ticker** con AutoGluon. Cada modelo estima la probabilidad de que su acción suba más de un 1% en los próximos 5 días. Después, las acciones se ordenan por esa probabilidad y se seleccionan las mejores oportunidades con la misma estrategia validada en backtest.

## Para qué sirve

El proyecto sirve para:

- Descargar datos históricos diarios de acciones.
- Procesar esos datos y crear features técnicas y de contexto de mercado.
- Entrenar modelos de clasificación por acción.
- Validar los modelos con backtests.
- Generar rankings de oportunidades ordenadas por probabilidad estimada de subida.
- Probar la estrategia en paper trading con dinero ficticio.

No es un sistema de inversión real ni una recomendación financiera. Es un entorno de experimentación para comprobar si la señal obtenida en backtest aguanta con datos nuevos.

## Estado actual

- Pipeline principal funcionando: ingesta, procesamiento, entrenamiento y backtest.
- Horizonte actual: 5 días.
- Target actual: `target_updown_t5`, que vale 1 si la acción sube más de un 1% en los próximos 5 días.
- Mejor resultado documentado: **+$55,490 (+454.90%)** desde $10,000 en backtest con estrategia top-3.
- Tickers con mejor señal confirmada: `MU`, `AMD`, `FCX`, `KLAC`, `AMAT`, `NVDA`, `CSCO`, `BAC`, `LRCX`.

## Estructura del proyecto

```text
.
├─ README.md
├─ requirements.txt
├─ docs/
│  ├─ baseline_resultados.md
│  ├─ proximos_pasos.md
│  ├─ Procesamiento.md
│  ├─ Entrenamiento.md
│  └─ ...
├─ scripts/
│  ├─ ingesta/
│  ├─ procesamiento/
│  ├─ entrenamiento/
│  └─ analisis/
├─ data/          # datos locales, no versionados
├─ modelos/       # modelos entrenados, no versionados
└─ evaluaciones/  # métricas y resultados, no versionados
```

## Partes principales

### Ingesta

Scripts en `scripts/ingesta/`.

- `ingesta_masiva.py`: descarga datos OHLCV diarios para el universo de acciones.
- `ingesta_individual.py`: descarga una acción concreta.
- `ingesta_market_data.py`: descarga datos de contexto de mercado como SPY y VIX.

Los datos se guardan localmente en `data/bronze/` y `data/market_data/`.

### Procesamiento

Scripts en `scripts/procesamiento/`.

- `procesado_clasificacion_5d.py`: genera el dataset Silver para clasificación a 5 días.
- `procesado_regresion.py`: pipeline anterior de regresión.
- `lectura_parquets.py`: utilidades para cargar datos Parquet.

El procesamiento crea features como retornos, volatilidad, RSI, MACD, Bollinger, distancia a máximos, volumen, SPY y VIX.

### Entrenamiento

Scripts en `scripts/entrenamiento/`.

- `autogluon_clasificacion_por_ticker.py`: entrena un modelo de clasificación por ticker.
- `autogluon_regresion_por_ticker.py`: entrenamiento anterior de regresión por ticker.
- `autogluon_regresion.py`: baseline global de regresión.

El modelo actual usa AutoGluon Tabular con split temporal y `tuning_data` para evitar leakage temporal.

### Análisis y backtest

Scripts en `scripts/analisis/`.

- `backtest_clasificacion.py`: evalúa los modelos de clasificación por ticker.
- `backtest_portfolio_topn.py`: simula una estrategia portfolio top-N ordenando por `P(sube)`.
- `backtest_periodos.py`: backtest usado en etapas anteriores de regresión.
- `analizar_feature_importance.py`: análisis de importancia de variables.

## Flujo general

```bash
# 1. Descargar acciones
python scripts/ingesta/ingesta_masiva.py

# 2. Descargar datos de mercado
python scripts/ingesta/ingesta_market_data.py

# 3. Generar dataset Silver de clasificación
python scripts/procesamiento/procesado_clasificacion_5d.py

# 4. Entrenar modelos por ticker
python scripts/entrenamiento/autogluon_clasificacion_por_ticker.py

# 5. Validar estrategia
python scripts/analisis/backtest_clasificacion.py --modo long_only
python scripts/analisis/backtest_portfolio_topn.py --top 3 --umbral 0.52
```

## Estrategia actual

La estrategia actual debe seguir exactamente la lógica validada en el backtest de clasificación:

- Cada modelo predice `P(sube)` para su ticker.
- Las acciones se ordenan de mayor a menor probabilidad.
- Se seleccionan las mejores oportunidades con estrategia top-N.
- El horizonte de operación es de 5 días.
- El paper trading debe replicar esta misma lógica, sin cambiar la estrategia hasta validarla primero.

## Prioridades de desarrollo

**Principales (siguiente fase):**

- `scripts/analisis/obtener_predicciones.py`: cargar modelos, pedir `P(sube)` del día y mostrar ranking completo.
- Registro de operaciones ficticias en Excel o CSV.
- Pipeline actual: ingesta, procesado, entrenamiento y backtest.

**Secundario:**

- Página web local para visualizar señales y resultados (opcional, más adelante).

## Documentación

- `docs/baseline_resultados.md`: resultados de experimentos y backtests.
- `docs/proximos_pasos.md`: tareas pendientes y líneas de mejora.
- `docs/Procesamiento.md`: explicación del procesamiento y features.
- `docs/Entrenamiento.md`: guía de entrenamiento.
- `docs/clasificacion_binaria.md`: explicación del enfoque de clasificación.

## Siguientes pasos

Los próximos pasos están en `docs/proximos_pasos.md`. Lo principal ahora es `obtener_predicciones.py` y el registro de operaciones ficticias. La experimentación con nuevos modelos y la web quedan para después.
