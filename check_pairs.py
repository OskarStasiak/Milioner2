import json
import time
import logging
from coinbase.rest import RESTClient

# Konfiguracja
with open('cdp_api_key.json', 'r') as f:
    api_keys = json.load(f)
    API_KEY = api_keys['api_key_id']
    API_SECRET = api_keys['api_key_secret']

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO)

def get_products():
    """Pobiera listę dostępnych par handlowych."""
    try:
        # Inicjalizacja klienta REST
        client = RESTClient(api_key=API_KEY, api_secret=API_SECRET)
        
        # Pobierz listę produktów
        products = client.get_products()
        
        if products and hasattr(products, 'products'):
            print("\n=== Dostępne pary handlowe ===")
            
            # Grupuj pary według waluty bazowej
            pairs_by_base = {}
            for product in products.products:
                base = product.base_currency_id
                quote = product.quote_currency_id
                if base not in pairs_by_base:
                    pairs_by_base[base] = []
                pairs_by_base[base].append(f"{base}-{quote}")
            
            # Wyświetl pary pogrupowane
            for base, pairs in sorted(pairs_by_base.items()):
                print(f"\n{base}:")
                for pair in sorted(pairs):
                    print(f"  - {pair}")
                    
            return products
        else:
            print("Błąd: Nie udało się pobrać listy produktów")
            return None
            
    except Exception as e:
        print(f"Błąd podczas pobierania par: {e}")
        return None

if __name__ == "__main__":
    get_products() 