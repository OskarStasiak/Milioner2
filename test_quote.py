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

def get_quote(from_account_id, to_account_id, from_amount, from_currency, to_currency):
    try:
        load_dotenv('production.env')
        api_key_name = os.getenv('COINBASE_API_KEY_NAME')
        api_key_secret = os.getenv('COINBASE_API_KEY_SECRET')
        if not api_key_name or not api_key_secret:
            log_message("Brak wymaganych zmiennych środowiskowych")
            return None
        payload = {
            "from_account_id": from_account_id,
            "to_account_id": to_account_id,
            "from_amount": from_amount,
            "from_currency": from_currency,
            "to_currency": to_currency
        }
        timestamp = str(int(time.time()))
        path = '/api/v3/brokerage/convert/quote'
        message = timestamp + 'POST' + path + json.dumps(payload)
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
        context = ssl._create_unverified_context()
        conn = http.client.HTTPSConnection("api.coinbase.com", context=context)
        conn.request("POST", path, json.dumps(payload), headers)
        response = conn.getresponse()
        data = response.read()
        if response.status == 200:
            result = json.loads(data.decode("utf-8"))
            log_message("Odpowiedź z API:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
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
    # UZUPEŁNIJ DANYMI SWOJEGO KONTA:
    from_account_id = "TWOJE_FROM_ACCOUNT_ID"
    to_account_id = "TWOJE_TO_ACCOUNT_ID"
    from_amount = "0.001"  # np. 0.001
    from_currency = "BTC"
    to_currency = "USD"
    log_message(f"Pobieram wycenę konwersji {from_amount} {from_currency} na {to_currency}...")
    get_quote(from_account_id, to_account_id, from_amount, from_currency, to_currency)

if __name__ == "__main__":
    main() 