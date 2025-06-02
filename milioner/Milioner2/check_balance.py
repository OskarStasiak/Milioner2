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

def check_balances():
    """Sprawdza salda konta używając bezpośrednio API Coinbase."""
    try:
        print("\n=== SPRAWDZANIE SALD KONTA ===")
        accounts = client.get_accounts().accounts
        
        print("\n{:<10} {:<15} {:<15}".format(
            "Waluta", "Dostępne", "Całkowite"
        ))
        print("-" * 45)
        
        for account in accounts:
            if hasattr(account, 'currency') and hasattr(account, 'available_balance'):
                currency = account.currency
                if currency in ['USDC', 'USD', 'BTC', 'ETH']:
                    available = float(account.available_balance.value)
                    total = float(account.total_balance.value) if hasattr(account, 'total_balance') else available
                    print("{:<10} {:<15.8f} {:<15.8f}".format(
                        currency,
                        available,
                        total
                    ))
        
        print("\n=========================")
    except Exception as e:
        print(f"Błąd podczas sprawdzania sald: {e}")

if __name__ == "__main__":
    check_balances() 