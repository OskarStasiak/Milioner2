import websocket
import json
import time
import logging
import os
from dotenv import load_dotenv
from coinbase.rest import RESTClient
from pathlib import Path

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Załaduj zmienne środowiskowe
load_dotenv(str(Path(__file__).parent / 'production.env'))

# Konfiguracja API
API_KEY = os.getenv('CDP_API_KEY_ID')
API_SECRET = os.getenv('CDP_API_KEY_SECRET')

# Inicjalizacja klienta REST
client = RESTClient(api_key=API_KEY, api_secret=API_SECRET)

def check_active_orders():
    """Sprawdza aktywne zlecenia na koncie."""
    try:
        print("\n=== SPRAWDZANIE AKTYWNYCH ZLECEŃ ===")
        orders = client.get_orders()
        
        if not orders.orders:
            print("Brak aktywnych zleceń")
            return
            
        print("\n{:<15} {:<10} {:<10} {:<15} {:<15}".format(
            "Para", "Typ", "Strona", "Cena", "Ilość"
        ))
        print("-" * 70)
        
        for order in orders.orders:
            print("{:<15} {:<10} {:<10} {:<15.8f} {:<15.8f}".format(
                order.product_id,
                order.order_type,
                order.side,
                float(order.price) if order.price else 0,
                float(order.size) if order.size else 0
            ))
            
    except Exception as e:
        print(f"Błąd podczas sprawdzania zleceń: {e}")

# URL WebSocket Coinbase Advanced Trade
WS_URL = "wss://advanced-trade-ws.coinbase.com"

# Funkcja obsługująca otrzymane wiadomości
def on_message(ws, message):
    try:
        data = json.loads(message)
        logging.info(f"Otrzymano wiadomość: {json.dumps(data, indent=2)}")
    except Exception as e:
        logging.error(f"Błąd podczas przetwarzania wiadomości: {e}")

# Funkcja obsługująca błędy
def on_error(ws, error):
    logging.error(f"Błąd WebSocket: {error}")

# Funkcja obsługująca zamknięcie połączenia
def on_close(ws, close_status_code, close_msg):
    logging.info(f"Zamknięto połączenie: {close_status_code} - {close_msg}")

# Funkcja obsługująca otwarcie połączenia
def on_open(ws):
    logging.info("Połączenie otwarte. Sprawdzam aktywne zlecenia...")
    check_active_orders()
    
    logging.info("Subskrybuję ticker na pary handlowe...")
    subscribe_message = {
        "type": "subscribe",
        "product_ids": ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "XRP-USD"],
        "channel": "ticker"
    }
    logging.info(f"Wysyłam subskrypcję: {json.dumps(subscribe_message, indent=2)}")
    ws.send(json.dumps(subscribe_message))

# Funkcja obsługująca ping
def on_ping(ws, message):
    logging.info("Otrzymano ping")

# Funkcja obsługująca pong
def on_pong(ws, message):
    logging.info("Otrzymano pong")

# Utworzenie połączenia WebSocket
ws = websocket.WebSocketApp(
    WS_URL,
    on_message=on_message,
    on_error=on_error,
    on_close=on_close,
    on_open=on_open,
    on_ping=on_ping,
    on_pong=on_pong
)

# Uruchomienie WebSocket w pętli
logging.info("Łączenie z WebSocket...")
ws.run_forever(
    ping_interval=20,
    ping_timeout=10,
    ping_payload='{"type":"ping"}',
    skip_utf8_validation=True
) 