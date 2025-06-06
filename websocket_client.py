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
        self.callback = callback  # Dodajemy callback
        self.ws = None
        self.connected = False
        self.subscribed_channels = set()  # Zbiór subskrybowanych kanałów
        self.last_reconnect_time = 0
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.min_reconnect_delay = 5  # Zmniejszone opóźnienie między próbami połączenia
        self.last_message_time = None
        self.message_count = 0
        self.last_ping = None
        self.last_pong = None
        self.reconnect_timer = None
        self.connection_lock = threading.Lock()
        
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
        
        self.WS_API_URL = "wss://advanced-trade-ws.coinbase.com"
        
        # Słownik do przechowywania danych dla każdej pary
        self.market_data = {}
        for product_id in self.trading_pairs:
            self.market_data[product_id] = {
                'order_book': {'bids': [], 'asks': []},
                'last_trade': None,
                'ticker': None
            }
    
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
    
    def on_message(self, ws, message):
        """Obsługa wiadomości WebSocket."""
        try:
            data = json.loads(message)
            logging.info(f"Otrzymano wiadomość: {json.dumps(data, indent=2)}")
            self.message_count += 1
            self.last_message_time = datetime.utcnow()
            
            # Obsługa ping/pong
            if data.get('type') == 'ping':
                self.last_ping = datetime.utcnow()
                ws.send(json.dumps({'type': 'pong'}))
                return
                
            if data.get('type') == 'pong':
                self.last_pong = datetime.utcnow()
                return
            
            # Obsługa różnych typów wiadomości
            if 'channel' in data:
                channel = data['channel']
                if channel == 'level2':
                    self._handle_level2(data)
                elif channel == 'ticker':
                    self._handle_ticker(data)
                elif channel == 'market_trades':
                    self._handle_trades(data)
                elif channel == 'status':
                    self._handle_status(data)
            
        except Exception as e:
            logging.error(f"Błąd podczas przetwarzania wiadomości WebSocket: {e}")
    
    def _handle_level2(self, data):
        """Obsługa danych order book."""
        try:
            product_id = data.get('product_id')
            if product_id in self.market_data:
                if 'bids' in data:
                    self.market_data[product_id]['order_book']['bids'] = data['bids']
                if 'asks' in data:
                    self.market_data[product_id]['order_book']['asks'] = data['asks']
                logging.debug(f"Zaktualizowano order book dla {product_id}")
                
                # Przekaż dane do głównego bota przez callback
                if self.callback:
                    self.callback(data)
                    
        except Exception as e:
            logging.error(f"Błąd podczas obsługi danych level2: {e}")
    
    def _handle_ticker(self, data):
        """Obsługa danych ticker."""
        try:
            product_id = data.get('product_id')
            if product_id in self.market_data:
                self.market_data[product_id]['ticker'] = data
                logging.debug(f"Zaktualizowano ticker dla {product_id}")
                
                # Przekaż dane do głównego bota przez callback
                if self.callback:
                    self.callback(data)
                    
        except Exception as e:
            logging.error(f"Błąd podczas obsługi danych ticker: {e}")
    
    def _handle_trades(self, data):
        """Obsługa danych o transakcjach."""
        try:
            product_id = data.get('product_id')
            if product_id in self.market_data:
                self.market_data[product_id]['last_trade'] = data
                logging.debug(f"Zaktualizowano ostatnią transakcję dla {product_id}")
                
                # Przekaż dane do głównego bota przez callback
                if self.callback:
                    self.callback(data)
                    
        except Exception as e:
            logging.error(f"Błąd podczas obsługi danych o transakcjach: {e}")
    
    def _handle_status(self, data):
        """Obsługa wiadomości o statusie."""
        try:
            if data.get('type') == 'error':
                logging.error(f"Błąd WebSocket: {data.get('message')}")
            elif data.get('type') == 'subscriptions':
                logging.info(f"Subskrypcje: {data.get('channels')}")
        except Exception as e:
            logging.error(f"Błąd podczas obsługi statusu: {e}")
    
    def on_error(self, ws, error):
        """Obsługa błędów WebSocket."""
        with self.connection_lock:
            logging.error(f"Błąd WebSocket: {error}")
            self.connected = False
            self._schedule_reconnect()
    
    def on_close(self, ws, close_status_code, close_msg):
        """Obsługa zamknięcia połączenia WebSocket."""
        with self.connection_lock:
            logging.info(f"Zamknięto połączenie WebSocket: {close_status_code} - {close_msg}")
            self.connected = False
            self.subscribed_channels.clear()
            self._schedule_reconnect()
    
    def on_open(self, ws):
        """Obsługa otwarcia połączenia WebSocket."""
        with self.connection_lock:
            logging.info("Otwarto połączenie WebSocket")
            self.connected = True
            self.reconnect_attempts = 0
            # Subskrybuj do kanałów po otwarciu połączenia
            self._subscribe_to_channels()
    
    def _subscribe_to_channels(self):
        """Subskrybuje do kanałów WebSocket dla każdej pary handlowej."""
        try:
            for product_id in self.trading_pairs:
                # Subskrybuj do kanału ticker
                subscribe_message = {
                    "type": "subscribe",
                    "product_ids": [product_id],
                    "channel": "ticker"
                }
                logging.info(f"Subskrybuję do kanału ticker dla {product_id}")
                self.ws.send(json.dumps(subscribe_message))
                time.sleep(0.2)  # Krótsze opóźnienie

                # Subskrybuj do kanału level2
                subscribe_message = {
                    "type": "subscribe",
                    "product_ids": [product_id],
                    "channel": "level2"
                }
                logging.info(f"Subskrybuję do kanału level2 dla {product_id}")
                self.ws.send(json.dumps(subscribe_message))
                time.sleep(0.2)

                # Subskrybuj do kanału market_trades
                subscribe_message = {
                    "type": "subscribe",
                    "product_ids": [product_id],
                    "channel": "market_trades"
                }
                logging.info(f"Subskrybuję do kanału market_trades dla {product_id}")
                self.ws.send(json.dumps(subscribe_message))
                time.sleep(0.2)

        except Exception as e:
            logging.error(f"Błąd podczas subskrybowania do kanałów: {str(e)}")
            raise
    
    def _run_forever(self):
        """Uruchamia WebSocket w pętli z obsługą błędów."""
        try:
            self.ws.run_forever(
                ping_interval=20,
                ping_timeout=10,
                ping_payload='{"type":"ping"}',
                skip_utf8_validation=True
            )
        except Exception as e:
            logging.error(f"Błąd w pętli WebSocket: {e}")
            self.connected = False
            self._schedule_reconnect()
    
    def _schedule_reconnect(self):
        """Planuje ponowne połączenie."""
        if self.reconnect_timer is not None:
            self.reconnect_timer.cancel()
        
        delay = self.min_reconnect_delay * (2 ** min(self.reconnect_attempts, 5))
        self.reconnect_timer = threading.Timer(delay, self._attempt_reconnect)
        self.reconnect_timer.daemon = True
        self.reconnect_timer.start()
    
    def _attempt_reconnect(self):
        """Próbuje ponownie nawiązać połączenie."""
        with self.connection_lock:
            try:
                if self.reconnect_attempts >= self.max_reconnect_attempts:
                    logging.error("Przekroczono maksymalną liczbę prób ponownego połączenia")
                    return
                
                self.reconnect_attempts += 1
                logging.info(f"Próba ponownego połączenia {self.reconnect_attempts}/{self.max_reconnect_attempts}")
                self.connect()
                
            except Exception as e:
                logging.error(f"Błąd podczas próby ponownego połączenia: {e}")
                self.connected = False
                self._schedule_reconnect()
    
    def connect(self):
        """Nawiązuje połączenie WebSocket."""
        with self.connection_lock:
            try:
                if self.ws is not None:
                    self.disconnect()
                
                # Przygotuj nagłówki
                headers = {
                    'User-Agent': 'Mozilla/5.0',
                    'Origin': 'https://advanced-trade-ws.coinbase.com'
                }
                
                # Utwórz połączenie WebSocket
                websocket.enableTrace(True)
                self.ws = websocket.WebSocketApp(
                    self.WS_API_URL,
                    header=headers,
                    on_message=self.on_message,
                    on_error=self.on_error,
                    on_close=self.on_close,
                    on_open=self.on_open,
                    on_ping=self.on_ping,
                    on_pong=self.on_pong
                )
                
                # Uruchom WebSocket w osobnym wątku
                self.ws_thread = threading.Thread(target=self._run_forever)
                self.ws_thread.daemon = True
                self.ws_thread.start()
                
                # Nie blokuj głównego wątku - pozwól na asynchroniczne połączenie
                logging.info("Rozpoczęto łączenie z WebSocket...")
                
            except Exception as e:
                logging.error(f"Błąd podczas nawiązywania połączenia WebSocket: {e}")
                self.connected = False
                self._schedule_reconnect()
                raise
    
    def disconnect(self):
        """Zamyka połączenie WebSocket."""
        with self.connection_lock:
            try:
                if self.reconnect_timer is not None:
                    self.reconnect_timer.cancel()
                    self.reconnect_timer = None
                
                if self.ws:
                    self.ws.close()
                    self.ws = None
                
                self.connected = False
                self.subscribed_channels.clear()
                
            except Exception as e:
                logging.error(f"Błąd podczas zamykania połączenia WebSocket: {e}")
    
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
            if not self.connected:
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

    def is_connected(self):
        """Sprawdza czy połączenie jest aktywne."""
        return self.connected and self.ws and self.ws.sock and self.ws.sock.connected

    def subscribe_to_ticker(self, product_id):
        """Subskrybuje do kanału ticker dla danej pary."""
        try:
            if not self.connected or not self.ws:
                logging.error("Brak połączenia WebSocket")
                return
            
            message = {
                "type": "subscribe",
                "product_ids": [product_id],
                "channel": "ticker"
            }
            
            self.ws.send(json.dumps(message))
            self.subscribed_channels.add(f"ticker_{product_id}")
            logging.info(f"Subskrybowano do kanału ticker dla {product_id}")
            time.sleep(0.5)  # Dodaj opóźnienie między subskrypcjami
            
        except Exception as e:
            logging.error(f"Błąd podczas subskrybowania do kanału ticker: {e}")
            raise

    def subscribe_to_level2(self, product_id):
        """Subskrybuje do kanału level2 dla danej pary."""
        try:
            if not self.connected or not self.ws:
                logging.error("Brak połączenia WebSocket")
                return
            
            message = {
                "type": "subscribe",
                "product_ids": [product_id],
                "channel": "level2"
            }
            
            self.ws.send(json.dumps(message))
            self.subscribed_channels.add(f"level2_{product_id}")
            logging.info(f"Subskrybowano do kanału level2 dla {product_id}")
            time.sleep(0.5)  # Dodaj opóźnienie między subskrypcjami
            
        except Exception as e:
            logging.error(f"Błąd podczas subskrybowania do kanału level2: {e}")
            raise

    def subscribe_to_market_trades(self, product_id):
        """Subskrybuje do kanału market_trades dla danej pary."""
        try:
            if not self.connected or not self.ws:
                logging.error("Brak połączenia WebSocket")
                return
            
            message = {
                "type": "subscribe",
                "product_ids": [product_id],
                "channel": "market_trades"
            }
            
            self.ws.send(json.dumps(message))
            self.subscribed_channels.add(f"market_trades_{product_id}")
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