import os
from dotenv import load_dotenv
from coinbase.rest import RESTClient
from pathlib import Path

# Załaduj zmienne środowiskowe
load_dotenv(str(Path(__file__).parent / 'production.env'))

# Konfiguracja API
API_KEY = os.getenv('CDP_API_KEY_ID')
API_SECRET = os.getenv('CDP_API_KEY_SECRET')

# Inicjalizacja klienta
client = RESTClient(api_key=API_KEY, api_secret=API_SECRET)

def get_transaction_history(product_id):
    """Pobiera historię transakcji dla danej pary."""
    try:
        print(f"\n=== HISTORIA TRANSAKCJI DLA {product_id} ===")
        print("Pobieranie danych...")
        
        fills = client.get_fills(product_id=product_id, limit=100)
        if not fills or not hasattr(fills, 'fills'):
            print("Brak historii transakcji")
            return
            
        if not fills.fills:
            print("Brak transakcji dla tej pary")
            return
            
        print(f"\nZnaleziono {len(fills.fills)} transakcji:")
        print("\n{:<20} {:<10} {:<15} {:<15} {:<15} {:<10}".format(
            "Data", "Typ", "Cena", "Ilość", "Wartość", "Opłata"
        ))
        print("-" * 90)
        
        for fill in fills.fills:
            try:
                data = getattr(fill, 'trade_time', '-')
                typ = 'KUPNO' if getattr(fill, 'side', '-') == 'BUY' else 'SPRZEDAŻ'
                cena = float(getattr(fill, 'price', 0))
                ilosc = float(getattr(fill, 'size', 0))
                wartosc = cena * ilosc
                oplata = getattr(fill, 'fee', '0')
                
                print("{:<20} {:<10} {:<15.8f} {:<15.8f} {:<15.2f} {:<10}".format(
                    str(data),
                    typ,
                    cena,
                    ilosc,
                    wartosc,
                    str(oplata)
                ))
            except Exception as e:
                print(f"Błąd przy przetwarzaniu transakcji: {e}")
                continue
                
        print("\n=========================")
    except Exception as e:
        print(f"Błąd podczas pobierania historii transakcji dla {product_id}: {e}")

if __name__ == "__main__":
    pairs = ["PEPE-USD", "ETH-USD", "USDC-USD"]
    for pair in pairs:
        get_transaction_history(pair) 