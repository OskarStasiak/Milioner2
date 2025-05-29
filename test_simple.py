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

def get_accounts(cursor=None):
    """Pobiera listę kont z obsługą paginacji."""
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
        if cursor:
            path += f'?cursor={cursor}'
            
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
            return json.loads(data.decode("utf-8"))
        else:
            log_message(f"Błąd API: {response.status}")
            log_message(f"Treść odpowiedzi: {data.decode('utf-8')}")
            return None
            
    except Exception as e:
        log_message(f"Wystąpił błąd: {str(e)}")
        return None
    finally:
        conn.close()

def display_account(account):
    """Wyświetla szczegóły konta."""
    log_message(f"\nKonto: {account.get('name', 'N/A')}")
    log_message(f"ID: {account.get('uuid', 'N/A')}")
    log_message(f"Waluta: {account.get('currency', 'N/A')}")
    log_message(f"Status: {'Aktywne' if account.get('active') else 'Nieaktywne'}")
    log_message(f"Typ: {account.get('type', 'N/A')}")
    log_message(f"Platforma: {account.get('platform', 'N/A')}")
    
    # Saldo dostępne
    balance = account.get('available_balance', {})
    if balance:
        log_message(f"Saldo dostępne: {balance.get('value', '0')} {balance.get('currency', 'N/A')}")
    
    # Saldo zablokowane
    hold = account.get('hold', {})
    if hold:
        log_message(f"Saldo zablokowane: {hold.get('value', '0')} {hold.get('currency', 'N/A')}")
    
    # Daty
    log_message(f"Utworzono: {account.get('created_at', 'N/A')}")
    log_message(f"Zaktualizowano: {account.get('updated_at', 'N/A')}")

def main():
    """Główna funkcja programu."""
    log_message("Pobieranie listy kont z API Coinbase...")
    
    all_accounts = []
    cursor = None
    
    while True:
        # Pobierz stronę kont
        response = get_accounts(cursor)
        if not response:
            break
            
        # Dodaj konta do listy
        accounts = response.get('accounts', [])
        if isinstance(accounts, list):
            all_accounts.extend(accounts)
        else:
            all_accounts.append(accounts)
            
        # Sprawdź czy jest następna strona
        if not response.get('has_next'):
            break
            
        # Pobierz cursor do następnej strony
        cursor = response.get('cursor')
        if not cursor:
            break
            
    # Wyświetl wszystkie konta
    log_message(f"\nZnaleziono {len(all_accounts)} kont:")
    for account in all_accounts:
        display_account(account)

if __name__ == "__main__":
    main() 