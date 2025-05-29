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
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}")

def display_permissions(permissions):
    """Wyświetla uprawnienia w czytelnym formacie."""
    if not permissions:
        return

    log_message("\n=== UPRAWNIENIA API ===")
    
    # Podstawowe uprawnienia
    log_message("\nPodstawowe uprawnienia:")
    log_message(f"Możliwość przeglądania: {'Tak' if permissions.get('can_view') else 'Nie'}")
    log_message(f"Możliwość handlu: {'Tak' if permissions.get('can_trade') else 'Nie'}")
    log_message(f"Możliwość transferu: {'Tak' if permissions.get('can_transfer') else 'Nie'}")
    
    # Informacje o portfelu
    log_message("\nInformacje o portfelu:")
    log_message(f"UUID portfela: {permissions.get('portfolio_uuid', 'N/A')}")
    log_message(f"Typ portfela: {permissions.get('portfolio_type', 'N/A')}")

def get_permissions():
    try:
        load_dotenv('production.env')
        api_key_name = os.getenv('COINBASE_API_KEY_NAME')
        api_key_secret = os.getenv('COINBASE_API_KEY_SECRET')
        
        if not api_key_name or not api_key_secret:
            log_message("Brak wymaganych zmiennych środowiskowych")
            return None

        timestamp = str(int(time.time()))
        method = 'GET'
        path = '/api/v3/brokerage/accounts'
        body = ''
        
        # Przygotowanie wiadomości do podpisu
        message = f"{timestamp}{method}{path}{body}"
        
        # Generowanie podpisu
        signature = hmac.new(
            api_key_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        # Przygotowanie nagłówków
        headers = {
            'Content-Type': 'application/json',
            'CB-ACCESS-KEY': api_key_name.split('/')[-1],
            'CB-ACCESS-SIGN': signature,
            'CB-ACCESS-TIMESTAMP': timestamp,
            'CB-ORG-ID': api_key_name.split('/')[1]
        }

        # Wyświetlenie informacji debugowych
        log_message("\n=== INFORMACJE DEBUGOWE ===")
        log_message(f"Timestamp: {timestamp}")
        log_message(f"Message: {message}")
        log_message(f"Signature: {signature}")
        log_message(f"Headers: {json.dumps(headers, indent=2)}")

        context = ssl._create_unverified_context()
        conn = http.client.HTTPSConnection("api.coinbase.com", context=context)
        conn.request(method, path, body, headers)
        response = conn.getresponse()
        data = response.read()

        if response.status == 200:
            result = json.loads(data.decode("utf-8"))
            display_permissions(result)
            return result
        else:
            log_message(f"Błąd API: {response.status}")
            log_message(f"Treść odpowiedzi: {data.decode('utf-8')}")
            return None
    except Exception as e:
        log_message(f"Wystąpił błąd: {str(e)}")
        return None
    finally:
        try:
            conn.close()
        except:
            pass

def main():
    log_message("Sprawdzanie uprawnień API...")
    get_permissions()

if __name__ == "__main__":
    main() 