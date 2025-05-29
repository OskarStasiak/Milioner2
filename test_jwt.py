import json
import jwt
import time
from websocket import WebSocketApp
import logging

# Konfiguracja
API_KEY = "organizations/7ffbb587-c89b-49a7-8e49-45186964fa12/apiKeys/cbbab9b8-f3d5-4c47-88cd-2d4b78e8cec4"
API_SECRET = """-----BEGIN EC PRIVATE KEY-----
MHcCAQEEIJ2BSdbB3sdyn223562f7NavQmf1So0c4vvjuVHVFYVToAoGCCqGSM49
AwEHoUQDQgAEF7k3tv1jt4/KCymlAcLz0KCmhRsUlm/pTP252tT+a736g/yQyrCV
oAvjZ88L4bJqZwW19KRfkIzV0Ay47ylO7A==
-----END EC PRIVATE KEY-----"""

# Konfiguracja logowania
logging.basicConfig(level=logging.DEBUG)

def generate_jwt():
    timestamp = int(time.time())
    expires = timestamp + 60
    
    header = {
        "alg": "ES256",
        "typ": "JWT"
    }
    
    payload = {
        "sub": API_KEY,
        "iat": timestamp,
        "exp": expires,
        "aud": ["retail_rest_api_pro"]
    }
    
    try:
        token = jwt.encode(payload, API_SECRET, algorithm="ES256", headers=header)
        print("\n=== JWT Token ===")
        print(f"Header: {json.dumps(header, indent=2)}")
        print(f"Payload: {json.dumps(payload, indent=2)}")
        print(f"Token: {token}")
        return token
    except Exception as e:
        print(f"Błąd generowania JWT: {e}")
        return None

def on_message(ws, message):
    print(f"\n[WS] Otrzymano wiadomość: {message}")

def on_error(ws, error):
    print(f"\n[WS] Błąd: {error}")

def on_close(ws, close_status_code, close_msg):
    print(f"\n[WS] Zamknięto połączenie: {close_status_code} - {close_msg}")

def on_open(ws):
    print("\n[WS] Połączono!")
    jwt_token = generate_jwt()
    if jwt_token:
        subscribe_message = {
            "type": "subscribe",
            "product_ids": ["BTC-USDC"],
            "channel": "ticker",
            "jwt": jwt_token
        }
        print(f"\n[WS] Wysyłam subskrypcję: {json.dumps(subscribe_message, indent=2)}")
        ws.send(json.dumps(subscribe_message))

def main():
    print("=== Test JWT i WebSocket ===")
    print(f"API Key: {API_KEY}")
    print(f"API Secret: {API_SECRET[:50]}...")
    
    # Generuj JWT
    jwt_token = generate_jwt()
    if not jwt_token:
        print("Nie udało się wygenerować JWT. Kończę działanie.")
        return
    
    # Połącz z WebSocket
    ws = WebSocketApp(
        "wss://advanced-trade-ws.coinbase.com",
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
        on_open=on_open
    )
    
    print("\n[WS] Próba połączenia z WebSocket...")
    ws.run_forever()

if __name__ == "__main__":
    main() 