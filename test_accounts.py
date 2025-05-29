import http.client
import json
import os
from dotenv import load_dotenv
import hmac
import hashlib
import time
from datetime import datetime
import ssl

def log_message(message):
    """Loguje wiadomość z timestampem."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}")

def get_accounts():
    """Pobiera listę kont."""
    try:
        # Ładowanie zmiennych środowiskowych
        load_dotenv('production.env')
        
        # Pobieranie kluczy API
        api_key_name = os.getenv('COINBASE_API_KEY_NAME')
        api_key_secret = os.getenv('COINBASE_API_KEY_SECRET')
        
        if not api_key_name or not api_key_secret:
            log_message("Brak wymaganych zmiennych środowiskowych")
            return None
            
        # Przygotowanie nagłówków
        timestamp = str(int(time.time()))
        path = '/api/v3/brokerage/accounts'
        message = timestamp + 'GET' + path
        
        signature = hmac.new(
            api_key_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        headers = {
            'Content-Type': 'application/json',
            'CB-ACCESS-KEY': api_key_name.split('/')[-1],
            'CB-ACCESS-SIGN': signature,
            'CB-ACCESS-TIMESTAMP': timestamp,
            'CB-ORG-ID': api_key_name.split('/')[1]
        }
        
        # Tworzenie kontekstu SSL
        context = ssl._create_unverified_context()
        
        # Tworzenie połączenia
        conn = http.client.HTTPSConnection("api.coinbase.com", context=context)
        
        # Wykonanie zapytania
        conn.request("GET", path, '', headers)
        
        # Pobranie odpowiedzi
        response = conn.getresponse()
        data = response.read()
        
        # Sprawdzenie statusu odpowiedzi
        if response.status == 200:
            result = json.loads(data.decode("utf-8"))
            log_message("Pobrano listę kont!")
            return result
        else:
            log_message(f"Błąd API: {response.status}")
            log_message(f"Treść odpowiedzi: {data.decode('utf-8')}")
            return None
            
    except Exception as e:
        log_message(f"Wystąpił błąd: {str(e)}")
        return None
    finally:
        conn.close()

def display_accounts(accounts):
    """Wyświetla informacje o kontach."""
    if not accounts or 'accounts' not in accounts:
        log_message("Brak kont do wyświetlenia")
        return
        
    log_message(f"\nZnaleziono {len(accounts['accounts'])} kont:")
    
    for account in accounts['accounts']:
        log_message("\nKonto:")
        log_message(f"ID: {account.get('uuid', 'N/A')}")
        log_message(f"Nazwa: {account.get('name', 'N/A')}")
        log_message(f"Waluta: {account.get('currency', 'N/A')}")
        
        # Saldo
        available = account.get('available_balance', {})
        log_message(f"Dostępne saldo: {available.get('value', '0')} {available.get('currency', 'N/A')}")
        
        hold = account.get('hold', {})
        log_message(f"Zablokowane środki: {hold.get('value', '0')} {hold.get('currency', 'N/A')}")
        
        # Status
        log_message(f"Status: {account.get('active', 'N/A')}")

def main():
    """Główna funkcja programu."""
    log_message("Pobieranie listy kont...")
    
    # Pobranie listy kont
    result = get_accounts()
    
    if result:
        display_accounts(result)
    else:
        log_message("Nie udało się pobrać listy kont")

if __name__ == "__main__":
    main() 