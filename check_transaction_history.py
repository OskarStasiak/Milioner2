import os
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
from coinbase.rest import RESTClient
import logging

# Konfiguracja loggera
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Załaduj zmienne środowiskowe
load_dotenv('production.env')

# Konfiguracja API
with open('cdp_api_key.json', 'r') as f:
    api_keys = json.load(f)
    API_KEY = api_keys['api_key_id']
    API_SECRET = api_keys['api_key_secret']

# Inicjalizacja klienta
client = RESTClient(api_key=API_KEY, api_secret=API_SECRET)

def get_transaction_history(product_id):
    """Pobiera historię transakcji dla danej pary."""
    try:
        print(f"\n=== HISTORIA TRANSAKCJI DLA {product_id} ===")
        print("Pobieranie danych...")
        
        # Pobierz świeczki z ostatnich 24 godzin
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=1)
        
        candles = client.get_candles(
            product_id=product_id,
            start=int(start_time.timestamp()),
            end=int(end_time.timestamp()),
            granularity="ONE_HOUR"
        )
        
        if not candles or not hasattr(candles, 'candles'):
            print("Brak danych historycznych")
            return
            
        print(f"\nZnaleziono {len(candles.candles)} świeczek:")
        print("\n{:<20} {:<15} {:<15} {:<15}".format(
            "Czas", "Cena", "Wolumen", "Wartość"
        ))
        print("-" * 70)
        
        for candle in candles.candles:
            try:
                time = datetime.fromtimestamp(int(candle.start))
                price = float(candle.close)
                volume = float(candle.volume)
                value = price * volume
                
                print("{:<20} {:<15.8f} {:<15.8f} {:<15.2f}".format(
                    str(time),
                    price,
                    volume,
                    value
                ))
            except Exception as e:
                print(f"Błąd przy przetwarzaniu świeczki: {e}")
                continue
                
        print("\n=========================")
    except Exception as e:
        print(f"Błąd podczas pobierania historii dla {product_id}: {e}")

def main():
    print("\n======================================================================")
    print("             HISTORIA TRANSAKCJI Z OSTATNICH 24 GODZIN                ")
    print("======================================================================")
    
    # Sprawdź pary handlowe
    pairs = ['ETH-USDC', 'BTC-USDC', 'SOL-USDC', 'DOGE-USDC', 'XRP-USDC']
    for pair in pairs:
        get_transaction_history(pair)

if __name__ == "__main__":
    main() 