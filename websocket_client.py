import json
import logging
import websocket
import threading
import time
import jwt
import hashlib
import os
from datetime import datetime, timedelta
import hmac
import base64
import ssl

class CoinbaseWebSocket:
    def __init__(self, api_key, api_secret, trading_pairs, callback=None):
        """
        Inicjalizacja klienta WebSocket dla Coinbase Advanced.
        
        Args:
            api_key (str): Klucz API w formacie "organizations/{org_id}/apiKeys/{key_id}"
            api_secret (str): Klucz prywatny w formacie PEM
            trading_pairs (list): Lista par handlowych do subskrypcji (np. ['ETH-USDC', 'BTC-USDC'])
            callback (function): Funkcja callback do przekazywania danych do głównego bota
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.trading_pairs = trading_pairs
        self.callback = callback
        self.ws = None
        self._is_connected = False
        self.subscription_queue = trading_pairs.copy()  # Inicjalizacja kolejki subskrypcji
        self.max_subscriptions = 3  # Maksymalna liczba równoczesnych subskrypcji
        self.current_subscriptions = 0
        self.last_reconnect_time = 0
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 20
        self.min_reconnect_delay = 5
        self.last_message_time = None
        self.message_count = 0
        self.last_ping = None
        self.last_pong = None
        self.reconnect_timer = None
        self.connection_lock = threading.Lock()
        self.last_reconnect_reset = datetime.utcnow()  # Czas ostatniego resetu
        self.reconnect_reset_interval = timedelta(hours=1)  # Reset co godzinę
        
        # Konfiguracja logowania
        logging.basicConfig(
            filename='websocket.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Kanały WebSocket
        self.CHANNEL_NAMES = {
            "level2": "level2",
            "user": "user",
            "tickers": "ticker",
            "ticker_batch": "ticker_batch",
            "status": "status",
            "market_trades": "market_trades",
            "candles": "candles",
        }
        
        self.WS_API_URL = "wss://ws-feed.exchange.coinbase.com"
        
        # Słownik do przechowywania danych dla każdej pary
        self.market_data = {}
        for product_id in self.trading_pairs:
            self.market_data[product_id] = {
                'order_book': {'bids': [], 'asks': []},
                'last_trade': None,
                'ticker': None
            }
            logging.info(f"Zainicjalizowano monitoring dla pary {product_id}")
    
    def _get_signature(self, timestamp, method, request_path, body=''):
        message = f"{timestamp}{method}{request_path}{body}"
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
        
    def _get_jwt(self):
        """Generuje JWT token dla autoryzacji WebSocket."""
        try:
            timestamp = int(time.time())
            expires = timestamp + 60  # Token ważny przez 60 sekund

            # Przygotuj nagłówek JWT
            header = {
                "alg": "ES256",
                "typ": "JWT"
            }

            # Przygotuj payload JWT zgodnie z dokumentacją Coinbase
            payload = {
                "sub": self.api_key,
                "iat": timestamp,
                "exp": expires,
                "aud": "retail_rest_api"
            }

            # Zakoduj nagłówek i payload
            encoded_header = base64.urlsafe_b64encode(json.dumps(header).encode()).rstrip(b'=').decode()
            encoded_payload = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b'=').decode()

            # Podpisz token
            message = f"{encoded_header}.{encoded_payload}"
            signature = hmac.new(
                self.api_secret.encode(),
                message.encode(),
                hashlib.sha256
            ).digest()
            encoded_signature = base64.urlsafe_b64encode(signature).rstrip(b'=').decode()

            # Połącz wszystkie części
            token = f"{encoded_header}.{encoded_payload}.{encoded_signature}"
            
            logging.info("Wygenerowano JWT token")
            return token

        except Exception as e:
            logging.error(f"Błąd podczas generowania JWT tokena: {str(e)}")
            return None
    
    @property
    def is_connected(self):
        """Sprawdza czy połączenie jest aktywne."""
        return self._is_connected and self.ws and self.ws.sock and self.ws.sock.connected

    def connect(self):
        """Nawiązanie połączenia WebSocket"""
        try:
            self.ws = websocket.WebSocketApp(
                self.WS_API_URL,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=self._on_open
            )
            self.ws.run_forever()
        except Exception as e:
            logging.error(f"Błąd połączenia WebSocket: {e}")
            self._is_connected = False

    def _on_open(self, ws):
        """Obsługa otwarcia połączenia"""
        self._is_connected = True
        logging.info("Połączenie WebSocket otwarte")
        try:
            # Subskrybuj wszystkie pary jednocześnie
            subscribe_message = {
                "type": "subscribe",
                "product_ids": self.trading_pairs,
                "channels": ["ticker"]
            }
            self.ws.send(json.dumps(subscribe_message))
            logging.info(f"Subskrybowano wszystkie pary: {self.trading_pairs}")
        except Exception as e:
            logging.error(f"Błąd subskrypcji: {e}")
            self._is_connected = False

    def _on_message(self, ws, message):
        """Obsługa otrzymanych wiadomości"""
        try:
            data = json.loads(message)
            
            if data.get('type') == 'subscriptions':
                logging.info(f"Potwierdzono subskrypcję: {data}")
            elif data.get('type') == 'error':
                logging.error(f"Błąd WebSocket: {data}")
            elif data.get('type') == 'ticker' and self.callback:
                self.callback(message)
                
        except Exception as e:
            logging.error(f"Błąd przetwarzania wiadomości: {e}")

    def _on_error(self, ws, error):
        """Obsługa błędów"""
        logging.error(f"Błąd WebSocket: {error}")
        self._is_connected = False

    def _on_close(self, ws, close_status_code, close_msg):
        """Obsługa zamknięcia połączenia"""
        logging.info("Połączenie WebSocket zamknięte")
        self._is_connected = False
        self.current_subscriptions = 0
        self.subscription_queue = self.trading_pairs.copy()  # Przywróć kolejkę

    def disconnect(self):
        """Zamknięcie połączenia"""
        if self.ws:
            self.ws.close()
            self._is_connected = False
            self.current_subscriptions = 0
            self.subscription_queue = self.trading_pairs.copy()

    def get_market_data(self, product_id):
        """Pobierz dane rynkowe dla danej pary."""
        return self.market_data.get(product_id, {})
    
    def get_order_book(self, product_id):
        """Pobierz order book dla danej pary."""
        return self.market_data.get(product_id, {}).get('order_book', {'bids': [], 'asks': []})
    
    def get_last_trade(self, product_id):
        """Pobierz ostatnią transakcję dla danej pary."""
        return self.market_data.get(product_id, {}).get('last_trade')
    
    def get_ticker(self, product_id):
        """Pobierz ticker dla danej pary."""
        return self.market_data.get(product_id, {}).get('ticker')

    def subscribe_to_channels(self, product_id):
        """Subskrybuj do kanałów dla danej pary handlowej."""
        try:
            if not self.is_connected:
                raise Exception("Brak połączenia WebSocket")
                
            # Pobierz nowy JWT
            jwt = self._get_jwt()
            
            # Subskrybuj do kanałów
            channels = ['ticker', 'level2', 'market_trades']
            for channel in channels:
                subscribe_message = {
                    "type": "subscribe",
                    "product_ids": [product_id],
                    "channel": channel,
                    "jwt": jwt
                }
                self.ws.send(json.dumps(subscribe_message))
                logging.info(f"Subskrybowano do kanału {channel} dla {product_id}")
                time.sleep(0.1)  # Małe opóźnienie między subskrypcjami
                
        except Exception as e:
            logging.error(f"Błąd podczas subskrybowania do kanałów dla {product_id}: {e}")
            raise

    def subscribe_to_ticker(self, product_id):
        """Subskrybuje do kanału ticker dla danej pary."""
        try:
            if not self.is_connected or not self.ws:
                logging.error("Brak połączenia WebSocket")
                return
            
            message = {
                "type": "subscribe",
                "product_ids": [product_id],
                "channel": "ticker"
            }
            
            self.ws.send(json.dumps(message))
            self.subscription_queue.append(product_id)
            logging.info(f"Subskrybowano do kanału ticker dla {product_id}")
            time.sleep(0.5)  # Dodaj opóźnienie między subskrypcjami
            
        except Exception as e:
            logging.error(f"Błąd podczas subskrybowania do kanału ticker: {e}")
            raise

    def subscribe_to_level2(self, product_id):
        """Subskrybuje do kanału level2 dla danej pary."""
        try:
            if not self.is_connected or not self.ws:
                logging.error("Brak połączenia WebSocket")
                return
            
            message = {
                "type": "subscribe",
                "product_ids": [product_id],
                "channel": "level2"
            }
            
            self.ws.send(json.dumps(message))
            self.subscription_queue.append(product_id)
            logging.info(f"Subskrybowano do kanału level2 dla {product_id}")
            time.sleep(0.5)  # Dodaj opóźnienie między subskrypcjami
            
        except Exception as e:
            logging.error(f"Błąd podczas subskrybowania do kanału level2: {e}")
            raise

    def subscribe_to_market_trades(self, product_id):
        """Subskrybuje do kanału market_trades dla danej pary."""
        try:
            if not self.is_connected or not self.ws:
                logging.error("Brak połączenia WebSocket")
                return
            
            message = {
                "type": "subscribe",
                "product_ids": [product_id],
                "channel": "market_trades"
            }
            
            self.ws.send(json.dumps(message))
            self.subscription_queue.append(product_id)
            logging.info(f"Subskrybowano do kanału market_trades dla {product_id}")
            time.sleep(0.5)  # Dodaj opóźnienie między subskrypcjami
            
        except Exception as e:
            logging.error(f"Błąd podczas subskrybowania do kanału market_trades: {e}")
            raise

    def on_ping(self, ws, message):
        """Obsługa ping."""
        try:
            self.last_ping = datetime.utcnow()
            ws.send(json.dumps({'type': 'pong'}))
        except Exception as e:
            logging.error(f"Błąd podczas obsługi ping: {e}")

    def on_pong(self, ws, message):
        """Obsługa pong."""
        try:
            self.last_pong = datetime.utcnow()
        except Exception as e:
            logging.error(f"Błąd podczas obsługi pong: {e}")

if __name__ == "__main__":
    # Przykład użycia
    API_KEY = "organizations/{org_id}/apiKeys/{key_id}"  # Zastąp swoim kluczem API
    API_SECRET = """-----BEGIN EC PRIVATE KEY-----
YOUR PRIVATE KEY
-----END EC PRIVATE KEY-----"""  # Zastąp swoim kluczem prywatnym
    
    # Zmiana par handlowych na USDC
    client = CoinbaseWebSocket(API_KEY, API_SECRET, ['ETH-USDC', 'BTC-USDC'])
    client.connect()
    
    try:
        while True:
            time.sleep(1)
            stats = client.get_stats()
            if stats['connected']:
                print(f"Połączono. Otrzymano {stats['message_count']} wiadomości.")
            else:
                print("Rozłączono. Próba ponownego połączenia...")
                client.connect()
                
    except KeyboardInterrupt:
        print("\nZamykanie klienta WebSocket...")
        client.disconnect() 