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

def get_balance_value(balance):
    if isinstance(balance, dict):
        return float(balance.get('value', 0))
    elif hasattr(balance, 'value'):
        return float(balance.value)
    else:
        return float(balance)

def check_all_balances():
    """Wyświetla szczegółowe salda wszystkich walut na koncie."""
    try:
        print("\n=== SZCZEGÓŁOWE SALDA KONTA ===")
        accounts = client.get_accounts().accounts
        print("\n{:<10} {:<15} {:<15} {:<15}".format(
            "Waluta", "Dostępne", "Całkowite", "Zablokowane"
        ))
        print("-" * 60)
        for account in accounts:
            if hasattr(account, 'currency') and hasattr(account, 'available_balance'):
                currency = account.currency
                available = get_balance_value(account.available_balance)
                total = get_balance_value(account.total_balance) if hasattr(account, 'total_balance') else available
                hold = get_balance_value(account.hold) if hasattr(account, 'hold') else 0
                print("{:<10} {:<15.8f} {:<15.8f} {:<15.8f}".format(
                    currency, available, total, hold
                ))
        print("\n=========================")
    except Exception as e:
        print(f"Błąd podczas sprawdzania sald: {e}")

if __name__ == "__main__":
    check_all_balances() 