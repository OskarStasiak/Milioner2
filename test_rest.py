import json
import time
import requests
import logging
import hmac
import hashlib

# Konfiguracja
API_KEY = "organizations/7ffbb587-c89b-49a7-8e49-45186964fa12/apiKeys/cbbab9b8-f3d5-4c47-88cd-2d4b78e8cec4"
API_SECRET = """-----BEGIN EC PRIVATE KEY-----
MHcCAQEEIJ2BSdbB3sdyn223562f7NavQmf1So0c4vvjuVHVFYVToAoGCCqGSM49
AwEHoUQDQgAEF7k3tv1jt4/KCymlAcLz0KCmhRsUlm/pTP252tT+a736g/yQyrCV
oAvjZ88L4bJqZwW19KRfkIzV0Ay47ylO7A==
-----END EC PRIVATE KEY-----"""

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO)

def get_signature(timestamp, method, request_path, body=''):
    message = f"{timestamp}{method}{request_path}{body}"
    signature = hmac.new(
        API_SECRET.encode('utf-8'),
        message.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    return signature

def get_accounts():
    timestamp = str(int(time.time()))
    method = "GET"
    path = "/api/v3/brokerage/accounts"
    signature = get_signature(timestamp, method, path)
    
    headers = {
        "Content-Type": "application/json",
        "CB-ACCESS-KEY": API_KEY,
        "CB-ACCESS-SIGN": signature,
        "CB-ACCESS-TIMESTAMP": timestamp,
        "CB-VERSION": "2023-11-15"
    }
    
    try:
        response = requests.get(
            "https://api.coinbase.com/api/v3/brokerage/accounts",
            headers=headers
        )
        print(f"\nStatus: {response.status_code}")
        print(f"Headers: {json.dumps(dict(response.headers), indent=2)}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Błąd podczas pobierania kont: {e}")

if __name__ == "__main__":
    get_accounts() 