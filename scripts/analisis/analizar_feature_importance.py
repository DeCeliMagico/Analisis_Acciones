"""
Analizar Feature Importance en los modelos per-ticker entrenados.
Ayuda a entender qué features son más importantes en cada ticker
y decide si explorar nuevas features.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from autogluon.tabular import TabularPredictor


# Configurar estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Tickers a analizar - se detectan automáticamente de los modelos disponibles
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELOS_DIR = PROJECT_ROOT / "modelos" / "regresion"


def detectar_tickers_disponibles() -> list:
    """Detecta automáticamente los tickers con modelos entrenados."""
    modelo_dirs = sorted(MODELOS_DIR.glob("Market_AI_Ticker_*"))
    
    if not modelo_dirs:
        return []
    
    # Extraer los tickers más recientes (ignorar duplicados con diferentes timestamps)
    tickers_set = set()
    for d in modelo_dirs:
        # Formato: Market_AI_Ticker_{TICKER}_{timestamp}
        parts = d.name.split("_")
        if len(parts) >= 4:
            ticker = parts[3]
            tickers_set.add(ticker)
    
    return sorted(list(tickers_set))


TOP_TICKERS = detectar_tickers_disponibles()

# Directorio de modelos - relativo al proyecto raíz


def cargar_feature_importance_ticker(ticker: str) -> dict | None:
    """Carga feature importance de un modelo per-ticker.
    
    Args:
        ticker: Símbolo del ticker
        
    Returns:
        dict con {feature: importance} o None si no existe modelo
    """
    
    # Buscar directorio del modelo para este ticker
    modelo_dirs = list(MODELOS_DIR.glob(f"Market_AI_Ticker_{ticker}_*"))
    
    if not modelo_dirs:
        print(f"❌ No se encontró modelo para {ticker}")
        return None
    
    modelo_path = modelo_dirs[0]  # Tomar el más reciente (si hay múltiples)
    
    try:
        # Cargar predictor
        predictor = TabularPredictor.load(str(modelo_path))
        
        # Extraer feature importance (usar feature_stage='transformed' no necesita dataset)
        importance = predictor.feature_importance(feature_stage='transformed')
        
        # Convertir a dict ordenado - manejar tanto Series como DataFrame
        importance_dict = {}
        
        if isinstance(importance, pd.Series):
            # Si es Series, convertir directamente asegurando valores numéricos
            for feature, value in importance.items():
                if isinstance(value, (int, float)):
                    importance_dict[feature] = float(value)
                else:
                    importance_dict[feature] = 0.0
        elif isinstance(importance, pd.DataFrame):
            # Si es DataFrame, tomar la primera columna y extraer valores numéricos
            col = importance.iloc[:, 0]
            for feature, value in col.items():
                if isinstance(value, (int, float, np.integer, np.floating)):
                    importance_dict[feature] = float(value)
                else:
                    importance_dict[feature] = 0.0
        else:
            # Fallback: intentar convertir diccionario directamente
            for k, v in importance.items():
                if isinstance(v, (int, float, np.integer, np.floating)):
                    importance_dict[k] = float(v)
                else:
                    importance_dict[k] = 0.0
        
        return importance_dict
        
    except Exception as e:
        print(f"❌ Error cargando modelo {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return None


def analizar_todos_tickers() -> dict:
    """Analiza feature importance para todos los tickers.
    
    Returns:
        dict {ticker: {feature: importance}}
    """
    
    resultados = {}
    
    print("="*70)
    print("ANALIZANDO FEATURE IMPORTANCE")
    print("="*70)
    
    for ticker in TOP_TICKERS:
        print(f"\n[{ticker}] Cargando feature importance...", end="")
        importance = cargar_feature_importance_ticker(ticker)
        
        if importance:
            resultados[ticker] = importance
            print(f" ✅ ({len(importance)} features)")
        else:
            print(f" ❌")
    
    return resultados


def crear_dataframe_importance(resultados: dict) -> tuple:
    """Convierte resultados a DataFrame para análisis.
    
    Args:
        resultados: dict {ticker: {feature: importance}}
        
    Returns:
        (DataFrame raw, DataFrame normalizado)
    """
    
    # Crear DataFrame - asegurar que todos los valores son numéricos
    df = pd.DataFrame(resultados).fillna(0)
    
    # Limpiar el índice - convertir None a string si es necesario
    df.index = df.index.astype(str)
    
    # Filtrar filas con "None" como nombre de feature
    df = df[df.index != 'None']
    
    # Convertir explícitamente a float64 para evitar problemas
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)
    
    # Normalizar por columna (cada ticker 0-1) usando min-max scaling
    df_norm = df.copy()
    for col in df_norm.columns:
        min_val = df_norm[col].min()
        max_val = df_norm[col].max()
        if max_val > min_val:
            df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
        else:
            df_norm[col] = 0
    
    return df, df_norm


def visualizar_importance(df_norm: pd.DataFrame):
    """Visualiza feature importance con heatmap y gráficos."""
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    
    # 1. Heatmap normalizado
    sns.heatmap(df_norm, annot=True, fmt='.3f', cmap='YlOrRd', 
                ax=axes[0], cbar_kws={'label': 'Importancia (normalizada)'})
    axes[0].set_title('Feature Importance por Ticker (Normalizado)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Ticker')
    axes[0].set_ylabel('Feature')
    
    # 2. Importancia promedio por feature
    importance_promedio = df_norm.mean(axis=1).sort_values(ascending=False)
    axes[1].barh(importance_promedio.index, importance_promedio.values, color='steelblue')
    axes[1].set_xlabel('Importancia Promedio (Normalizada)')
    axes[1].set_title('Importancia Promedio de Features Across All Tickers', fontsize=14, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar figura en la raíz del proyecto
    output_path = PROJECT_ROOT / "evaluaciones" / "feature_importance_analisis.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Gráfico guardado: {output_path}")
    
    plt.close()  # Cerrar para no bloquear ejecución


def main():
    """Pipeline completo de análisis."""
    
    # 1. Analizar feature importance
    resultados = analizar_todos_tickers()
    
    if not resultados:
        print("\n❌ No se encontraron modelos para analizar")
        return
    
    # 2. Crear DataFrames
    df, df_norm = crear_dataframe_importance(resultados)
    
    print(f"\n✅ Datos procesados para {len(resultados)} tickers")
    # Filtrar None del índice antes de convertir a string
    features = [str(f) for f in df.index if f is not None and str(f) != 'None']
    print(f"   Features: {', '.join(features)}")
    
    # 3. Visualizar
    visualizar_importance(df_norm)
    
    print("\n" + "="*70)
    print("✅ ANÁLISIS COMPLETADO")
    print("="*70)


if __name__ == "__main__":
    main()
