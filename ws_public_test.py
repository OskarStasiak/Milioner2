import websocket
import threading
import time
import logging
import json

def on_message(ws, message):
    print(f"ODEBRANO WIADOMOŚĆ: {message}")
    logging.info(f"ODEBRANO WIADOMOŚĆ: {message}")

def on_error(ws, error):
    print(f"BŁĄD: {error}")
    logging.error(f"BŁĄD: {error}")

def on_close(ws, close_status_code, close_msg):
    print(f"Zamknięto połączenie: {close_status_code} - {close_msg}")
    logging.info(f"Zamknięto połączenie: {close_status_code} - {close_msg}")

def on_open(ws):
    print("Połączenie otwarte. Wysyłam subskrypcję...")
    logging.info("Połączenie otwarte. Wysyłam subskrypcję...")
    subscribe_message = {
        "type": "subscribe",
        "product_ids": ["BTC-USDC"],
        "channel": "ticker"
    }
    ws.send(json.dumps(subscribe_message))
    print("Subskrypcja wysłana.")
    logging.info("Subskrypcja wysłana.")

if __name__ == "__main__":
    logging.basicConfig(filename='ws_public_test.log', level=logging.INFO)
    ws_url = "wss://advanced-trade-ws.coinbase.com"
    ws = websocket.WebSocketApp(
        ws_url,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
        on_open=on_open
    )
    wst = threading.Thread(target=ws.run_forever)
    wst.daemon = True
    wst.start()
    try:
        for _ in range(60):  # 60 sekund
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    ws.close()
    print("Koniec testu.") 