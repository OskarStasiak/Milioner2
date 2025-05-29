import os
import requests
from dotenv import load_dotenv
import json
from datetime import datetime

def log_message(message):
    """Loguje wiadomość z timestampem."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}")

def test_api_connection():
    """Testuje połączenie z API Coinbase."""
    try:
        # Ładowanie zmiennych środowiskowych
        load_dotenv('production.env')
        
        # Pobieranie kluczy API
        api_key_name = os.getenv('COINBASE_API_KEY_NAME')
        api_key_secret = os.getenv('COINBASE_API_KEY_SECRET')
        
        if not api_key_name or not api_key_secret:
            log_message("Brak wymaganych zmiennych środowiskowych")
            return
            
        # Przygotowanie nagłówków
        headers = {
            'Content-Type': 'application/json',
            'CB-ACCESS-KEY': api_key_name.split('/')[-1],
            'CB-ORG-ID': api_key_name.split('/')[1]
        }
        
        # Test połączenia - pobieranie listy kont
        response = requests.get(
            'https://api.coinbase.com/api/v3/brokerage/accounts',
            headers=headers
        )
        
        # Sprawdzenie odpowiedzi
        if response.status_code == 200:
            data = response.json()
            log_message("Połączenie udane!")
            log_message(f"Liczba kont: {len(data.get('accounts', []))}")
            
            # Wyświetl szczegóły kont
            for account in data.get('accounts', []):
                log_message(f"\nKonto: {account.get('name', 'N/A')}")
                log_message(f"ID: {account.get('uuid', 'N/A')}")
                log_message(f"Waluta: {account.get('currency', 'N/A')}")
                log_message(f"Status: {'Aktywne' if account.get('active') else 'Nieaktywne'}")
                
                # Saldo
                balance = account.get('available_balance', {})
                if balance:
                    log_message(f"Saldo: {balance.get('value', '0')} {balance.get('currency', 'N/A')}")
        else:
            log_message(f"Błąd API: {response.status_code}")
            log_message(f"Treść odpowiedzi: {response.text}")
            
    except Exception as e:
        log_message(f"Wystąpił błąd: {str(e)}")

if __name__ == "__main__":
    log_message("Testowanie połączenia z API Coinbase...")
    test_api_connection() 