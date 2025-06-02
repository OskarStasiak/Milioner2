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

def check_transaction_history(limit=20):
    """Wyświetla historię ostatnich transakcji na koncie."""
    try:
        print("\n=== HISTORIA OSTATNICH TRANSAKCJI ===")
        accounts = client.get_accounts().accounts
        for account in accounts:
            if hasattr(account, 'currency'):
                currency = account.currency
                print(f"\n--- {currency} ---")
                try:
                    txs = client.get_account_transactions(account.id)
                    if not txs.transactions:
                        print("Brak transakcji.")
                        continue
                    for tx in txs.transactions[:limit]:
                        print(f"Typ: {tx.type}, Kwota: {tx.amount}, Status: {tx.status}, Data: {tx.created_at}")
                except Exception as e:
                    print(f"Błąd pobierania transakcji: {e}")
        print("\n=========================")
    except Exception as e:
        print(f"Błąd podczas pobierania historii: {e}")

if __name__ == "__main__":
    check_transaction_history() 