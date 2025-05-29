import json
from coinbase.rest import RESTClient
import logging
import http.client
import hmac
import hashlib
import time
import ssl

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Wczytaj klucze API z pliku
with open('cdp_api_key.json', 'r') as f:
    api_keys = json.load(f)
    API_KEY = api_keys['api_key_id']
    API_SECRET = api_keys['api_key_secret']

# Wyłącz weryfikację SSL dla celów testowych
context = ssl._create_unverified_context()
conn = http.client.HTTPSConnection("api.coinbase.com", context=context)

# Przygotuj nagłówki z autoryzacją
timestamp = str(int(time.time()))
message = f"{timestamp}GET/api/v3/brokerage/accounts"
signature = hmac.new(
    API_SECRET.encode('utf-8'),
    message.encode('utf-8'),
    hashlib.sha256
).hexdigest()

headers = {
    'Content-Type': 'application/json',
    'CB-ACCESS-KEY': API_KEY,
    'CB-ACCESS-SIGN': signature,
    'CB-ACCESS-TIMESTAMP': timestamp,
    'CB-VERSION': '2023-11-15',
    'Accept': 'application/json'
}

# Pobierz listę wszystkich kont
conn.request("GET", "/api/v3/brokerage/accounts", '', headers)
response = conn.getresponse()
data = response.read().decode("utf-8")

print("Status:", response.status)
print("Response:", data)

# Jeśli mamy konta, pobierz szczegóły pierwszego konta
if response.status == 200:
    accounts = json.loads(data)
    if accounts.get('accounts'):
        first_account = accounts['accounts'][0]
        account_uuid = first_account.get('uuid')
        
        if account_uuid:
            # Pobierz szczegóły konkretnego konta
            message = f"{timestamp}GET/api/v3/brokerage/accounts/{account_uuid}"
            signature = hmac.new(
                API_SECRET.encode('utf-8'),
                message.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            headers['CB-ACCESS-SIGN'] = signature
            
            conn.request("GET", f"/api/v3/brokerage/accounts/{account_uuid}", '', headers)
            response = conn.getresponse()
            data = response.read().decode("utf-8")
            
            print("\nAccount Details:")
            print("Status:", response.status)
            print("Response:", data)

conn.close()

def check_balance(rest_client):
    """Sprawdza saldo konta."""
    try:
        response = rest_client.get_accounts()
        accounts = response['accounts']
        print("\n=== SALDA KONT ===")
        for account in accounts:
            currency = account['currency']
            available = float(account['available_balance']['value'])
            hold = float(account['hold']['value'])
            total = available + hold
            
            if total > 0:
                print(f"\nWaluta: {currency}")
                print(f"  Dostępne: {available}")
                print(f"  Zablokowane: {hold}")
                print(f"  Łącznie: {total}")
                print("-------------------")
        
        return accounts
    except Exception as e:
        print(f"Błąd podczas sprawdzania salda: {str(e)}")
        return None

def check_orders(rest_client):
    """Sprawdza aktywne zlecenia."""
    try:
        response = rest_client.get_orders()
        orders = response.get('orders', [])
        print("\n=== AKTYWNE ZLECENIA ===")
        
        if not orders:
            print("Brak aktywnych zleceń")
            return
            
        for order in orders:
            order_id = order['order_id']
            product_id = order['product_id']
            side = order['side']
            status = order['status']
            size = order['base_size']
            
            print(f"\nZlecenie {order_id}:")
            print(f"  Para: {product_id}")
            print(f"  Typ: {side}")
            print(f"  Status: {status}")
            print(f"  Wielkość: {size}")
            print("-------------------")
        
        return orders
    except Exception as e:
        print(f"Błąd podczas sprawdzania zleceń: {str(e)}")
        return None

def cancel_all_orders(rest_client):
    """Anuluje wszystkie aktywne zlecenia."""
    try:
        response = rest_client.list_orders()
        orders = response.orders if hasattr(response, 'orders') else []
        open_order_ids = [order['order_id'] for order in orders if order['status'] == 'OPEN']
        print("\n=== ANULOWANIE ZLECEŃ ===")
        if not open_order_ids:
            print("Brak aktywnych zleceń do anulowania.")
            return
        try:
            rest_client.cancel_orders(open_order_ids)
            print(f"Anulowano zlecenia: {open_order_ids}")
        except Exception as e:
            print(f"Błąd podczas anulowania zleceń: {str(e)}")
        print("Gotowe.")
    except Exception as e:
        print(f"Błąd podczas pobierania/anulowania zleceń: {str(e)}")

def main():
    # Inicjalizacja klienta REST
    rest_client = RESTClient(api_key=API_KEY, api_secret=API_SECRET, verbose=True)
    
    # Sprawdź salda
    check_balance(rest_client)
    
    # Anuluj wszystkie zlecenia
    cancel_all_orders(rest_client)

if __name__ == "__main__":
    main() 