import json
import logging
from websocket import WebSocketApp
import threading
import time
import jwt

API_KEY = "organizations/7ffbb587-c89b-49a7-8e49-45186964fa12/apiKeys/cbbab9b8-f3d5-4c47-88cd-2d4b78e8cec4"
API_SECRET = """-----BEGIN EC PRIVATE KEY-----\nMHcCAQEEIJ2BSdbB3sdyn223562f7NavQmf1So0c4vvjuVHVFYVToAoGCCqGSM49\nAwEHoUQDQgAEF7k3tv1jt4/KCymlAcLz0KCmhRsUlm/pTP252tT+a736g/yQyrCV\noAvjZ88L4bJqZwW19KRfkIzV0Ay47ylO7A==\n-----END EC PRIVATE KEY-----\n"""

WS_API_URL = "wss://advanced-trade-ws.coinbase.com"

logging.basicConfig(level=logging.INFO)

def get_jwt():
    timestamp = int(time.time())
    expires = timestamp + 60
    header = {"alg": "ES256", "typ": "JWT"}
    payload = {
        "sub": API_KEY,
        "iat": timestamp,
        "exp": expires,
        "aud": ["retail_rest_api_pro"]
    }
    token = jwt.encode(payload, API_SECRET, algorithm="ES256", headers=header)
    return token

def on_message(ws, message):
    print("[MESSAGE]", message)

def on_error(ws, error):
    print("[ERROR]", error)

def on_close(ws, close_status_code, close_msg):
    print(f"[CLOSE] {close_status_code} {close_msg}")

def on_open(ws):
    print("[OPEN] Połączono z WebSocket!")
    jwt_token = get_jwt()
    subscribe_message = {
        "type": "subscribe",
        "product_ids": ["BTC-USD"],
        "channel": "ticker",
        "jwt": jwt_token
    }
    ws.send(json.dumps(subscribe_message))
    print("[SEND] Subskrypcja wysłana")

def run_ws():
    ws = WebSocketApp(
        WS_API_URL,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
        on_open=on_open
    )
    ws.run_forever()

if __name__ == "__main__":
    run_ws() 