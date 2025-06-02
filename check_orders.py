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

def check_active_orders():
    """Sprawdza aktywne zlecenia na koncie."""
    try:
        print("\n=== SPRAWDZANIE AKTYWNYCH ZLECEŃ ===")
        orders = client.list_orders()
        
        if not orders:
            print("Brak aktywnych zleceń")
            return
            
        print("\n{:<15} {:<10} {:<10} {:<15} {:<15}".format(
            "Para", "Typ", "Strona", "Cena", "Ilość"
        ))
        print("-" * 70)
        
        for order in orders:
            print("{:<15} {:<10} {:<10} {:<15.8f} {:<15.8f}".format(
                order.product_id,
                order.order_type,
                order.side,
                float(order.price) if order.price else 0,
                float(order.size) if order.size else 0
            ))
            
    except Exception as e:
        print(f"Błąd podczas sprawdzania zleceń: {e}")

if __name__ == "__main__":
    check_active_orders() 