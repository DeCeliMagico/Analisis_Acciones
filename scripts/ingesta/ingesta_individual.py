import requests
import pandas as pd
import os
from pathlib import Path

def obtener_datos_accion(symbol):
    #URL de la api de Yahoo Finance para obtener los datos históricos de la acción seleccionada
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=max"

    #Llamada a la API
    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
    if response.status_code == 200:
        data = response.json()
        
        if 'chart' in data and 'result' in data['chart'] and len(data['chart']['result']) > 0 and len(data['chart']['result'][0]['indicators']['quote']) > 0:
            result = data['chart']['result'][0]
            quote = result['indicators']['quote'][0]
            df = pd.DataFrame({
                'ts_event_utc': pd.to_datetime(result['timestamp'], unit='s', utc=True),
                'open': quote['open'],
                'high': quote['high'],
                'low': quote['low'],
                'close': quote['close'],
                'volume': quote['volume'],
                'symbol': symbol
            })
            return df
        else:
            print("No se encontraron datos para el símbolo proporcionado.")
            return None
    else:
        print(f"Error al obtener datos: {response.status_code}")
        return None


def limpieza_minima(df):
    # Eliminar filas con valores nulos
    df = df.dropna()
    df = df.sort_values(by='ts_event_utc')
    return df


def df_to_parquet(df, symbol):
    bronze_dir = Path(__file__).resolve().parents[2] / "data" / "bronze"
    bronze_dir.mkdir(parents=True, exist_ok=True)

    output_path = bronze_dir / f"{symbol}_1d.parquet"
    df.to_parquet(output_path, index=False)

    print(f"Datos de {symbol} guardados en formato Parquet.")
    print(f"Filas guardadas: {len(df)}")
    print(f"Archivo guardado: {output_path}")



def guardar_datos_accion(symbol):
    df = obtener_datos_accion(symbol)
    if df is not None:
        df = limpieza_minima(df)
        df_to_parquet(df, symbol)


def main():
    guardar_datos_accion("AAPL")

if __name__ == "__main__":
    main()