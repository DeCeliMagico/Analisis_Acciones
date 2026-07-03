# Baseline — Resultados antes del Grupo 1 de mejoras
**Fecha:** 03-07-2026  
**Features:** ret_1d, ret_3d, gap_prop, range_norm, price_vs_ma20, volatility_5d, volatility_20d, rsi, macd, volume, vol_ma_20 (17 features con temporales de AutoGluon)  
**Modelos:** AutoGluon regresión por ticker (WeightedEnsemble_L2)

---

## ultimos resultados

(venv) PS D:\Analisis_Acciones> python .\scripts\analisis\backtest_periodos.py --modo long_short
==========================================================================================
BACKTEST PAPER TRADING (periodo TEST, out-of-sample)
==========================================================================================
Modo: long_short | Capital inicial: $10,000 | Comision: 0.10% por lado
Horizonte: 5 dias | test (15% final temporal, out-of-sample)
Período custom: Default (últimos 15%)

Ticker   Fecha Inicio Fecha Fin    Filas   
----------------------------------------
BAC      2019-07-01 13:30:00+00:00 2026-06-24 13:30:00+00:00 1755    
CSCO     2021-01-07 14:30:00+00:00 2026-06-24 13:30:00+00:00 1371    
F        2019-07-01 13:30:00+00:00 2026-06-24 13:30:00+00:00 1755    
GOOG     2023-03-15 13:30:00+00:00 2026-06-24 13:30:00+00:00 822     
KLAC     2019-08-12 13:30:00+00:00 2026-06-24 13:30:00+00:00 1726    
MSFT     2020-06-05 13:30:00+00:00 2026-06-24 13:30:00+00:00 1520    
NFLX     2022-11-09 14:30:00+00:00 2026-06-24 13:30:00+00:00 907     
NVDA     2022-05-12 13:30:00+00:00 2026-06-24 13:30:00+00:00 1032    
ORCL     2020-06-05 13:30:00+00:00 2026-06-24 13:30:00+00:00 1520    
TSLA     2024-02-01 14:30:00+00:00 2026-06-24 13:30:00+00:00 600     

Ticker    Trades    Win%   Ret.Modelo      Ret.B&H    MaxDD      Capital
------------------------------------------------------------------------------------------
BAC          292  54.5%      -51.05%       98.67%    75.7% $       489
CSCO         228  53.1%      -26.38%      169.12%    55.1% $       736
F            292  49.0%      -21.40%       35.02%    57.9% $       786
GOOG         137  57.7%       30.48%      265.36%    44.3% $     1,305
KLAC         287  57.5%      580.69%     1663.54%    36.7% $     6,807
MSFT         253  51.0%       36.34%       99.39%    36.1% $     1,363
NFLX         151  43.0%      -56.18%      172.13%    61.9% $       438
NVDA         172  60.5%     1184.95%     1094.24%    27.2% $    12,850
ORCL         253  46.2%      -66.19%      197.47%    79.6% $       338
TSLA         100  53.0%      200.97%      100.11%    32.7% $     3,010
------------------------------------------------------------------------------------------
TOTAL       2165              181.22%                       $    28,122

---

## Conclusión baseline
- R² negativo en 9 de 10 tickers → señal muy débil
- Solo NVDA y TSLA baten al buy & hold
- Features técnicos clásicos insuficientes → siguiente paso: Grupo 1 (momentum largo plazo + Bollinger + dist_52w + OBV)

---
---

# Grupo 1 — Resultados tras añadir momentum largo plazo + Bollinger + dist_52w_high + OBV
**Fecha:** 03-07-2026  
**Features nuevos:** ret_10d, ret_20d, ret_60d, bb_position, dist_52w_high, obv_ratio  
**Total features:** 23 features con temporales de AutoGluon

---

## Métricas de dirección (test out-of-sample)

| Ticker | Acc. Dirección | Correlación | Rel.RMSE |
|--------|---------------|-------------|----------|
| NVDA   | 60.32%        | 0.0080      | 1.0066   |
| CSCO   | 56.59%        | -0.0385     | 1.0168   |
| GOOG   | 56.22%        | 0.0189      | 1.0103   |
| MSFT   | 54.28%        | 0.0499      | 1.0267   |
| BAC    | 53.95%        | 0.0458      | 0.9993   |
| KLAC   | 53.05%        | 0.0542      | 1.0333   |
| NFLX   | 51.72%        | 0.0903      | 0.9960   |
| ORCL   | 51.68%        | -0.0099     | 1.3335   |
| F      | 51.34%        | 0.0613      | 1.0240   |
| TSLA   | 48.06%        | 0.1630      | 0.9899   |

## Backtest long_only

| Ticker | Trades | Win%  | Ret. Modelo | Ret. B&H   | MaxDD  |
|--------|--------|-------|-------------|------------|--------|
| BAC    | 263    | 54.8% | -2.11%      | +111.27%   | 48.1%  |
| CSCO   | 202    | 58.4% | +121.36%    | +166.30%   | 38.2%  |
| F      | 233    | 48.1% | +5.51%      | +52.96%    | 56.5%  |
| GOOG   | 126    | 57.1% | +94.89%     | +224.92%   | 24.2%  |
| KLAC   | 247    | 57.9% | +758.15%    | +1405.17%  | 31.9%  |
| MSFT   | 246    | 51.2% | -16.73%     | +81.19%    | 32.7%  |
| NFLX   | 134    | 50.7% | +89.35%     | +146.28%   | 36.8%  |
| NVDA   | 163    | 57.1% | +308.05%    | +1210.13%  | 43.9%  |
| ORCL   | 159    | 56.0% | +28.76%     | +181.95%   | 48.4%  |
| TSLA   | 94     | 41.5% | -6.95%      | +113.36%   | 51.9%  |
| **TOTAL** | **1867** | — | **+138.03%** | — | — |

**Capital final portfolio:** $23,803 (desde $10,000)

## Backtest long_short

| Ticker | Trades | Win%  | Ret. Modelo | Ret. B&H   | MaxDD  |
|--------|--------|-------|-------------|------------|--------|
| BAC    | 286    | 50.3% | -44.48%     | +111.27%   | 67.9%  |
| CSCO   | 222    | 55.4% | +32.34%     | +166.30%   | 58.5%  |
| F      | 286    | 49.7% | +51.40%     | +52.96%    | 53.3%  |
| GOOG   | 131    | 55.0% | +46.05%     | +224.92%   | 28.2%  |
| KLAC   | 281    | 46.6% | -68.58%     | +1405.17%  | 79.5%  |
| MSFT   | 247    | 52.2% | -10.59%     | +81.19%    | 45.2%  |
| NFLX   | 145    | 49.0% | +49.44%     | +146.28%   | 34.5%  |
| NVDA   | 166    | 59.6% | +589.38%    | +1210.13%  | 43.9%  |
| ORCL   | 247    | 48.6% | +29.43%     | +181.95%   | 64.6%  |
| TSLA   | 94     | 41.5% | -6.95%      | +113.36%   | 51.9%  |
| **TOTAL** | **2105** | — | **+66.74%** | — | — |

**Capital final portfolio:** $16,674 (desde $10,000)

## Conclusión Grupo 1
- Mejoras: NFLX (+3.4pp acc.), MSFT (+4pp), CSCO, GOOG ligeramente
- Empeoramientos: TSLA -6.6pp (cae a 48%, peor que azar), KLAC -4.8pp
- Long_short portfolio: $28,122 → $16,674 (peor)
- Long_only portfolio: $23,803 (sin baseline anterior para comparar)
- Conclusión: señal insuficiente, los técnicos clásicos ya están arbitrados
- Siguiente paso: Grupo 2 (SPY + VIX como contexto de mercado)
