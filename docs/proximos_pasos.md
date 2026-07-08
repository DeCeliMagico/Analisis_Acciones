---

## Estado actual (06-07-2026)

Pipeline de clasificación funcionando con los siguientes scripts:
1. `procesado_clasificacion_5d.py` → Silver con target_updown_t5 (sube >1% en 5d)
2. `autogluon_clasificacion_por_ticker.py` → 25 tickers, 180s, tuning_data temporal
3. `backtest_clasificacion.py` → evaluación por ticker (EV/trade, win%, avg win/loss)
4. `backtest_portfolio_topn.py` → portfolio top-N por P(sube), solo tickers con EV>0

**Mejor resultado conseguido:** $55,490 (+454.90%) con 9 tickers top (180s, 25 modelos entrenados)
**Tickers buenos confirmados:** MU, AMD, FCX, KLAC, AMAT, NVDA, CSCO, BAC, LRCX
**Comando portfolio:** `python backtest_portfolio_topn.py --top 3 --umbral 0.52 --tickers MU,AMD,FCX,KLAC,AMAT,NVDA,CSCO,BAC,LRCX`

---

## Próximos pasos — modelo

### Experimentación y exploración de la estructura del modelo

El setup actual (AutoGluon tabular, un modelo por ticker, clasificación binaria 5d) funciona,
pero la señal tiene techo. Hay que explorar si otras arquitecturas o enfoques de ML mejoran
la calidad predictiva sin introducir leakage temporal.

**Líneas a investigar:**

1. **Redes neuronales**
   - AutoGluon ya entrena NNs internamente, pero con poco tiempo (~60-100s/ticker).
   - Investigar si es rentable crear una red neuronal dedicada, por ejemplo con Keras, u otras
   alternativas que permitan mejorar los resultados.


2. **Otros tipos de ML**
   - Modelos secuenciales: LSTM, Transformer ligero sobre ventanas de precios.
   - Modelos de boosting con tuning manual (XGBoost/LightGBM fuera de AutoGluon).
   - Modelos probabilísticos (Gaussian Process, quantile regression) para estimar incertidumbre.
   - Ensemble híbrido: combinar clasificación tabular + señal de momentum simple.

3. **Nota**
   - Mantener siempre split temporal (train/valid/test) — no repetir el error del Run 2
     (480s + medium_quality sin tuning_data → leakage).
   - Comparar con la misma métrica: AUC, EV/trade y retorno del portfolio top-3 en test.

**Criterio de éxito:** solo adoptar un cambio si mejora resultados. Si no supera el baseline actual (+454.90% en backtest- clasificacion , estrategia top N acciones de todos los modelos),
no se despliega.

### Flujo práctico para paper trading (prioridad principal)

Crear las piezas necesarias para probar el sistema en la realidad con dinero ficticio, replicando exactamente la lógica del backtest con datos nuevos posteriores al entrenamiento.

**Tareas principales:**

1. **`obtener_predicciones.py`**
   - Crear `scripts/analisis/obtener_predicciones.py`.
   - Cargar el último Silver de clasificación.
   - Cargar todos los modelos entrenados en `modelos/clasificacion/`.
   - Pedir a cada modelo su `P(sube)` para la última fecha disponible.
   - Mostrar todos los tickers ordenados de mejor a peor, sin ocultar ninguno.
   - Marcar cuáles entrarían en la estrategia top-N del backtest (misma lógica, sin cambios).

2. **Registro de operaciones ficticias (Excel o CSV)**
   - Guardar fecha de entrada, acción, probabilidad, capital, comisión, precio de entrada, fecha de salida, precio de salida y resultado (retorno y capital resultante).
   - Mantener el horizonte de 5 días para que coincida con el target `target_updown_t5`.
   - Registrar manualmente las operaciones que se abren tras ver el ranking de `obtener_predicciones.py`.

3. **Validación previa con backtest**
   - Tras reentrenar, ejecutar `backtest_clasificacion.py` y `backtest_portfolio_topn.py`.
   - Comparar contra el baseline de +454.90% antes de usar esos modelos en paper trading.

**Tareas secundarias (más adelante):**

4. **Análisis posterior**
   - Cuando haya suficientes operaciones ficticias, analizar resultados con gráficas.
   - Medir evolución del capital, win rate, drawdown, retorno medio y comparación contra buy & hold.

5. **Página web local (opcional)**
   - Idea secundaria, más de ocio/producto.
   - Podría servir para ver ranking de señales, operaciones abiertas y evolución del capital ficticio.
   - No es necesaria para validar el modelo; primero conviene el flujo con scripts y Excel/CSV.
