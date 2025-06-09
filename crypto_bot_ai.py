import os
import sys
import json
import time
import logging
import threading
from datetime import datetime
from websocket_client import CoinbaseWebSocket
import numpy as np
import tensorflow as tf
import traceback
import requests
import hmac
import hashlib
import base64

class CryptoBotAI:
    def __init__(self, trading_pairs=None):
        """Inicjalizacja bota"""
        self.trading_pairs = trading_pairs or ['BTC-USD', 'ETH-USD', 'SOL-USD']
        self.ws = None
        self.ws_connected = False
        self.is_running = False
        self.logger = logging.getLogger('CryptoBotAI')
        
        # Zmienne do obsługi ponownego połączenia
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 5
        
        # Inicjalizacja danych rynkowych
        self.market_data = {pair: [] for pair in self.trading_pairs}
        
        # Wczytaj zapisane dane jeśli istnieją
        self.load_market_data()
        
        # Upewnij się, że katalog data istnieje
        os.makedirs('data', exist_ok=True)
        
        # Konfiguracja logowania
        self.setup_logging()
        self.logger.info("=== INICJALIZACJA BOTA ===")
        
        # Konfiguracja API
        self.api_key = os.getenv('COINBASE_API_KEY_NAME')
        self.api_secret = os.getenv('COINBASE_API_KEY_SECRET')
        self.base_url = "https://api.exchange.coinbase.com"
        
        # Parametry handlowe
        self.max_position_size = float(os.getenv('MAX_POSITION_SIZE', 0.05))
        self.trade_value_usdc = float(os.getenv('TRADE_VALUE_USDC', 50))
        self.stop_loss_percent = float(os.getenv('STOP_LOSS_PERCENT', 1))
        self.take_profit_percent = float(os.getenv('TAKE_PROFIT_PERCENT', 2))
        
        self.logger.info(f"Bot zainicjalizowany - będzie działał na parach: {', '.join(self.trading_pairs)}")
        
    def setup_logging(self):
        """Konfiguracja systemu logowania"""
        try:
            # Tworzenie katalogu na logi jeśli nie istnieje
            if not os.path.exists('logs'):
                os.makedirs('logs')
            
            # Konfiguracja loggera
            self.logger = logging.getLogger('CryptoBotAI')
            self.logger.setLevel(logging.INFO)
            
            # Usuwanie istniejących handlerów
            for handler in self.logger.handlers[:]:
                self.logger.removeHandler(handler)
            
            # Handler do pliku głównego
            file_handler = logging.FileHandler('logs/bot.log', mode='a')
            file_handler.setLevel(logging.INFO)
            file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
            
            # Handler do konsoli
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
            
            # Handler do pliku błędów
            error_handler = logging.FileHandler('logs/error.log', mode='a')
            error_handler.setLevel(logging.ERROR)
            error_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
            error_handler.setFormatter(error_formatter)
            self.logger.addHandler(error_handler)
            
        except Exception as e:
            print(f"Błąd podczas konfiguracji logowania: {e}")
            sys.exit(1)

    def connect_websocket(self):
        """Nawiązanie połączenia WebSocket z automatycznym reconnectem"""
        self.logger.info("=== PRÓBA POŁĄCZENIA Z WEBSOCKET ===")
        
        while self.reconnect_attempts < self.max_reconnect_attempts:
            try:
                self.logger.info(f"Próba połączenia z WebSocket (próba {self.reconnect_attempts + 1}/{self.max_reconnect_attempts})")
                
                # Inicjalizacja WebSocket
                self.ws = CoinbaseWebSocket(
                    api_key=os.getenv('COINBASE_API_KEY_NAME'),
                    api_secret=os.getenv('COINBASE_API_KEY_SECRET'),
                    trading_pairs=self.trading_pairs,
                    callback=self.update_market_data
                )
                
                # Uruchomienie WebSocket w osobnym wątku
                ws_thread = threading.Thread(target=self.ws.connect, daemon=True)
                ws_thread.start()
                
                # Czekamy na połączenie
                time.sleep(2)
                
                # Sprawdzamy czy połączenie jest aktywne
                if self.ws and self.ws.is_connected:
                    self.logger.info("Połączono z WebSocket pomyślnie!")
                    self.ws_connected = True
                    self.reconnect_attempts = 0
                    return True
                else:
                    raise Exception("Nie udało się nawiązać połączenia WebSocket")
                    
            except Exception as e:
                self.logger.error(f"Błąd połączenia z WebSocket: {e}")
                self.reconnect_attempts += 1
                if self.reconnect_attempts < self.max_reconnect_attempts:
                    self.logger.info(f"Ponowna próba za {self.reconnect_delay} sekund...")
                    time.sleep(self.reconnect_delay)
                else:
                    self.logger.error("Przekroczono maksymalną liczbę prób połączenia")
                    return False
        return False

    def on_error(self, error):
        """Obsługa błędów WebSocket"""
        self.logger.error(f"Błąd WebSocket: {error}")
        self.ws_connected = False
        self.is_running = False

    def on_close(self):
        """Obsługa zamknięcia połączenia WebSocket"""
        self.logger.info("Połączenie WebSocket zamknięte")
        self.ws_connected = False
        self.is_running = False

    def update_market_data(self, message):
        """Aktualizacja danych rynkowych"""
        try:
            data = json.loads(message)
            if 'type' in data and data['type'] == 'ticker':
                pair = data['product_id']
                if pair in self.trading_pairs:
                    price = float(data['price'])
                    volume_24h = float(data['volume_24h'])
                    
                    # Dodaj nowe dane
                    self.market_data[pair].append({
                        'price': price,
                        'volume_24h': volume_24h,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    # Ogranicz liczbę próbek do 1000
                    if len(self.market_data[pair]) > 1000:
                        self.market_data[pair] = self.market_data[pair][-1000:]
                    
                    # Zapisz dane co 100 próbek
                    if len(self.market_data[pair]) % 100 == 0:
                        self.save_market_data()
                    
                    # Loguj dane
                    self.logger.info("=== DANE RYNKOWE ===")
                    self.logger.info(f"Para: {pair}")
                    self.logger.info(f"Cena: {price} USD")
                    self.logger.info(f"Wolumen 24h: {volume_24h}")
                    self.logger.info(f"Liczba zebranych próbek: {len(self.market_data[pair])}")
                    self.logger.info("===================")
                    
        except Exception as e:
            self.logger.error(f"Błąd podczas aktualizacji danych: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")

    def save_market_data(self):
        """Zapisuje dane rynkowe do pliku"""
        try:
            for pair in self.trading_pairs:
                if self.market_data[pair]:
                    with open(f'data/market_data_{pair}.json', 'w') as f:
                        json.dump(self.market_data[pair], f)
                    self.logger.info(f"Zapisano {len(self.market_data[pair])} próbek dla {pair}")
        except Exception as e:
            self.logger.error(f"Błąd podczas zapisywania danych: {e}")

    def load_market_data(self):
        """Wczytuje dane rynkowe z pliku"""
        try:
            for pair in self.trading_pairs:
                data_file = f'data/market_data_{pair}.json'
                if os.path.exists(data_file):
                    with open(data_file, 'r') as f:
                        self.market_data[pair] = json.load(f)
                    self.logger.info(f"Wczytano {len(self.market_data[pair])} próbek dla {pair}")
        except Exception as e:
            self.logger.error(f"Błąd podczas wczytywania danych: {e}")

    def stop_bot(self):
        """Bezpieczne zatrzymanie bota"""
        try:
            self.logger.info("=== ZATRZYMYWANIE BOTA ===")
            self.is_running = False
            
            # Zatrzymanie WebSocket
            if self.ws:
                self.logger.info("Zatrzymywanie WebSocket...")
                self.ws.disconnect()
                self.ws = None
            
            # Zapisz dane przed zatrzymaniem
            self.save_market_data()
            
            self.logger.info("Bot został zatrzymany pomyślnie")
            return True
        except Exception as e:
            self.logger.error(f"Błąd podczas zatrzymywania bota: {e}")
            return False

    def handle_command(self, command):
        """Obsługa komend"""
        command = command.strip().lower()
        
        commands = {
            'help': self.show_help,
            'status': self.check_status,
            'stop': self.stop_bot,
            'trenowanie': self.train_model,
            'przewidywanie': self.test_predictions,
            'saldo': self.check_balances
        }
        
        if command in commands:
            self.logger.info(f"=== PRZEKAZUJĘ KOMENDĘ DO OBSŁUGI: {command} ===")
            return commands[command]()
        else:
            self.logger.warning(f"Nieznane polecenie: {command}")
            self.show_help()
            return False

    def check_status(self):
        """Sprawdzenie statusu bota"""
        status = {
            "running": self.is_running,
            "websocket_connected": self.ws_connected,
            "trading_pairs": self.trading_pairs
        }
        self.logger.info(f"Status bota: {status}")
        return True

    def start_bot(self):
        """Uruchomienie bota"""
        if not self.is_running:
            self.is_running = True
            self.logger.info("Bot uruchomiony")
            return True
        return False

    def show_help(self):
        """Wyświetlenie pomocy"""
        help_text = """
        Dostępne polecenia:
        - help      - wyświetla tę pomoc
        - status    - wyświetla status bota
        - stop      - zatrzymuje bota
        - trenowanie - rozpoczyna trenowanie modelu
        - przewidywanie - testuje przewidywania modelu
        - saldo     - wyświetla salda konta
        """
        self.logger.info(help_text)
        return True

    def check_balances(self):
        """Sprawdza salda konta."""
        try:
            self.logger.info("\n=== SPRAWDZANIE SALD KONTA ===")
            accounts = self.client.get_accounts().accounts
            
            print("\n{:<10} {:<15} {:<15} {:<15}".format(
                "Waluta", "Dostępne", "Całkowite", "Zablokowane"
            ))
            print("-" * 60)
            
            for account in accounts:
                if hasattr(account, 'currency') and hasattr(account, 'available_balance'):
                    currency = account.currency
                    if currency in ['USDC', 'USD', 'BTC', 'ETH']:
                        available = float(account.available_balance.value)
                        total = float(account.total_balance.value) if hasattr(account, 'total_balance') else available
                        hold = float(account.hold.value) if hasattr(account, 'hold') else 0
                        print("{:<10} {:<15.8f} {:<15.8f} {:<15.8f}".format(
                            currency, available, total, hold
                        ))
            
            self.logger.info("\n=========================")
            return True
        except Exception as e:
            self.logger.error(f"Błąd podczas sprawdzania sald: {e}")
            return False

    def train_model(self):
        """Trenowanie modelu na zebranych danych"""
        self.logger.info("=== ROZPOCZYNAM TRENOWANIE MODELU ===")
        
        # Sprawdź czy mamy wystarczająco danych
        for pair in self.trading_pairs:
            samples = len(self.market_data[pair])
            self.logger.info(f"Sprawdzam dane dla {pair}: {samples} próbek")
            if samples < 100:
                self.logger.warning(f"Za mało danych dla {pair} do trenowania (potrzeba min. 100, mamy {samples})")
                return
        
        # Przygotuj dane do trenowania
        pair_data = self.prepare_data_for_training()
        if not pair_data:
            return False
        
        # Trenuj model dla każdej pary
        for pair, data in pair_data.items():
            self.logger.info(f"=== TRENOWANIE MODELU DLA {pair} ===")
        
        # Podziel dane na zbiór treningowy i testowy
            split = int(len(data['X']) * 0.8)
            X_train, X_test = data['X'][:split], data['X'][split:]
            y_train, y_test = data['y'][:split], data['y'][split:]
        
        # Stwórz model
        model = tf.keras.Sequential([
                tf.keras.layers.LSTM(32, input_shape=(10, 2), return_sequences=True,
                                   kernel_regularizer=tf.keras.regularizers.l2(0.01)),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.LSTM(16,
                                   kernel_regularizer=tf.keras.regularizers.l2(0.01)),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(8, activation='relu',
                                    kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        self.logger.info("=== MODEL SKOMPILOWANY ===")
        
        # Dodaj early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Trenuj model
        self.logger.info("=== ROZPOCZYNAM TRENOWANIE ===")
        history = model.fit(
            X_train, y_train,
            epochs=100,  # Zwiększamy maksymalną liczbę epok
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Zapisz model
        model.save(f'models/crypto_model_{pair}.h5')
        self.logger.info(f"=== MODEL DLA {pair} ZAPISANY ===")
        
        # Zapisz historię treningu
        with open(f'models/training_history_{pair}.json', 'w') as f:
            json.dump(history.history, f)
        self.logger.info(f"=== HISTORIA TRENINGU DLA {pair} ZAPISANA ===")
        
        # Zapisz statystyki normalizacji
        with open(f'models/normalization_stats_{pair}.json', 'w') as f:
            json.dump(data['stats'], f)
        self.logger.info(f"=== STATYSTYKI NORMALIZACJI DLA {pair} ZAPISANE ===")
        
        # Zapisz informacje o ostatnim treningu
        training_info = {
            'last_training': datetime.now().isoformat(),
            'samples_used': len(data['X']),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'final_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1])
        }
        with open(f'models/training_info_{pair}.json', 'w') as f:
            json.dump(training_info, f)
        self.logger.info(f"=== INFORMACJE O TRENINGU DLA {pair} ZAPISANE ===")
        
        self.logger.info("=== TRENOWANIE ZAKOŃCZONE POMYŚLNIE ===")
        return True

    def analyze_trend(self, symbol: str, timeframe: str = '1h') -> dict:
        """
        Analizuje trend dla danego symbolu na podstawie danych historycznych.
        
        Args:
            symbol: Symbol handlowy (np. 'BTC-USD')
            timeframe: Okres czasowy (np. '1h', '4h', '1d')
            
        Returns:
            dict: Słownik zawierający informacje o trendzie
        """
        try:
            # Pobierz dane historyczne
            historical_data = self.market_data[symbol]
            
            if len(historical_data) < 24:  # Minimum 24 świeczki
                return {'trend': 'neutral', 'strength': 0}
            
            # Oblicz średnie kroczące
            close_prices = [float(d['price']) for d in historical_data[-24:]]
            sma_20 = sum(close_prices[-20:]) / 20
            sma_50 = sum(close_prices[-50:]) / 50 if len(close_prices) >= 50 else sma_20
            
            # Oblicz RSI
            rsi = self.calculate_rsi(close_prices)
            
            # Oblicz wolumen
            volumes = [float(d['volume_24h']) for d in historical_data[-24:]]
            avg_volume = sum(volumes[-20:]) / 20
            current_volume = volumes[-1]
            
            # Określ trend
            trend = 'neutral'
            strength = 0
            
            if sma_20 > sma_50:
                trend = 'bullish'
                strength += 1
            elif sma_20 < sma_50:
                trend = 'bearish'
                strength -= 1
            
            if rsi > 70:
                trend = 'bearish'
                strength -= 1
            elif rsi < 30:
                trend = 'bullish'
                strength += 1
            
            if current_volume > avg_volume * 1.5:
                if close_prices[-1] > close_prices[-2]:
                    strength += 1
                else:
                    strength -= 1
            
            return {
                'trend': trend,
                'strength': strength,
                'rsi': rsi,
                'volume_ratio': current_volume / avg_volume,
                'sma_20': sma_20,
                'sma_50': sma_50
            }
            
        except Exception as e:
            self.logger.error(f"Błąd podczas analizy trendu dla {symbol}: {str(e)}")
            return {'trend': 'neutral', 'strength': 0}

    def make_trading_decision(self, symbol: str, current_price: float, predicted_price: float) -> dict:
        """
        Podejmuje decyzję handlową na podstawie przewidywanej zmiany ceny i analizy technicznej.
        
        Args:
            symbol: Symbol pary handlowej
            current_price: Aktualna cena
            predicted_price: Przewidywana cena
            
        Returns:
            dict: Decyzja handlowa
        """
        try:
            # Oblicz przewidywaną zmianę ceny
            price_change = ((predicted_price - current_price) / current_price) * 100
            
            # Pobierz dane historyczne
            historical_data = self.market_data[symbol]
            if not historical_data:
                return {'action': 'hold', 'reason': 'brak danych historycznych'}
            
            # Oblicz wskaźniki techniczne
            prices = [float(candle['price']) for candle in historical_data]
            volumes = [float(candle['volume_24h']) for candle in historical_data]
            
            # Analiza trendu
            trend_analysis = self.analyze_trend(symbol)
            
            # Analiza wolumenu
            volume_analysis = self.analyze_volume(volumes)
            
            # Analiza zmienności
            volatility_analysis = self.calculate_volatility(prices)
            
            # Dostosuj próg na podstawie wskaźników
            base_threshold = float(os.getenv('PRICE_THRESHOLD_BUY', 5.0))
            
            # Zmniejsz próg jeśli trend jest silny
            if trend_analysis['strength'] > 0.7:
                base_threshold *= 0.7
            # Zwiększ próg jeśli zmienność jest wysoka
            if volatility_analysis['is_high_volatility']:
                base_threshold *= 1.3
            
            # Sprawdź warunki dla sygnału kupna
            if (price_change > base_threshold and 
                trend_analysis['trend'] == 'bullish' and
                volume_analysis['is_volume_increasing']):
                
                return {
                    'action': 'buy',
                    'reason': f'silny sygnał kupna: zmiana {price_change:.2f}%, trend {trend_analysis["trend"]}, wolumen {volume_analysis["volume_trend"]}'
                }
                
            # Sprawdź warunki dla sygnału sprzedaży
            if (price_change < -base_threshold and 
                trend_analysis['trend'] == 'bearish' and
                volume_analysis['is_volume_increasing']):
                
                return {
                    'action': 'sell',
                    'reason': f'silny sygnał sprzedaży: zmiana {price_change:.2f}%, trend {trend_analysis["trend"]}, wolumen {volume_analysis["volume_trend"]}'
                }
                
            return {
                'action': 'hold',
                'reason': f'brak silnego sygnału: zmiana {price_change:.2f}%, trend {trend_analysis["trend"]}, wolumen {volume_analysis["volume_trend"]}'
            }
            
        except Exception as e:
            self.logger.error(f"Błąd podczas podejmowania decyzji handlowej: {str(e)}")
            return {'action': 'hold', 'reason': f'błąd: {str(e)}'}

    def _get_signature(self, timestamp, method, request_path, body=''):
        """Generuje podpis dla żądania API."""
        message = f"{timestamp}{method}{request_path}{body}"
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature

    def _request(self, method, endpoint, params=None, data=None):
        """Wykonuje żądanie do API."""
        try:
            url = f"{self.base_url}{endpoint}"
            timestamp = str(int(time.time()))
            
            # Przygotuj nagłówki
            headers = {
                'CB-ACCESS-KEY': self.api_key,
                'CB-ACCESS-SIGN': self._get_signature(timestamp, method, endpoint, json.dumps(data) if data else ''),
                'CB-ACCESS-TIMESTAMP': timestamp,
                'Content-Type': 'application/json'
            }
            
            # Wykonaj żądanie
            response = requests.request(
                method,
                url,
                headers=headers,
                params=params,
                json=data
            )
            
            # Sprawdź odpowiedź
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            self.logger.error(f"Błąd podczas wykonywania żądania API: {e}")
            raise

    def get_account_balance(self, currency):
        """Pobiera saldo konta dla danej waluty."""
        try:
            accounts = self._request('GET', '/accounts')
            for account in accounts:
                if account['currency'] == currency:
                    return float(account['available'])
            return 0
        except Exception as e:
            self.logger.error(f"Błąd podczas pobierania salda: {e}")
            return 0

    def execute_trade(self, pair, decision, thresholds):
        """Wykonanie transakcji na podstawie decyzji handlowej"""
        try:
            if decision == "BUY":
                self.logger.info(f"=== WYKONYWANIE ZLECENIA KUPNA DLA {pair} ===")
                self.logger.info(f"Cena kupna: {thresholds['buy']:.2f} USD")
                self.logger.info(f"Cena sprzedaży: {thresholds['sell']:.2f} USD")
                
                # Sprawdź saldo USD
                usd_balance = self.get_account_balance('USD')
                if usd_balance < self.trade_value_usdc:
                    self.logger.warning(f"Za mało środków na koncie USD: {usd_balance:.2f} < {self.trade_value_usdc}")
                    return
                
                # Oblicz rozmiar zlecenia
                size = self.trade_value_usdc / thresholds['buy']
                
                # Złóż zlecenie
                order = self._request('POST', '/orders', data={
                    'product_id': pair,
                    'side': 'buy',
                    'type': 'market',
                    'size': str(size)
                })
                
                self.logger.info(f"Złożono zlecenie kupna: {order}")
                
            elif decision == "SELL":
                self.logger.info(f"=== WYKONYWANIE ZLECENIA SPRZEDAŻY DLA {pair} ===")
                self.logger.info(f"Cena kupna: {thresholds['buy']:.2f} USD")
                self.logger.info(f"Cena sprzedaży: {thresholds['sell']:.2f} USD")
                
                # Sprawdź saldo kryptowaluty
                crypto = pair.split('-')[0]
                crypto_balance = self.get_account_balance(crypto)
                if crypto_balance <= 0:
                    self.logger.warning(f"Brak {crypto} do sprzedaży")
                    return
                
                # Złóż zlecenie
                order = self._request('POST', '/orders', data={
                    'product_id': pair,
                    'side': 'sell',
                    'type': 'market',
                    'size': str(crypto_balance)
                })
                
                self.logger.info(f"Złożono zlecenie sprzedaży: {order}")
                
            else:
                self.logger.info(f"=== BRAK TRANSAKCJI DLA {pair} ===")
                
        except Exception as e:
            self.logger.error(f"Błąd podczas wykonywania transakcji: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")

    def test_predictions(self):
        """Testowanie przewidywań modelu na ostatnich danych"""
        try:
            self.logger.info("=== TESTUJĘ PRZEWIDYWANIA MODELU ===")
            
            for pair in self.trading_pairs:
                # Sprawdź czy mamy model dla tej pary
                model_path = f'models/crypto_model_{pair}.h5'
                stats_path = f'models/normalization_stats_{pair}.json'
                data_path = f'data/market_data_{pair}.json'
                
                if not os.path.exists(model_path) or not os.path.exists(stats_path):
                    self.logger.warning(f"Brak modelu lub statystyk dla {pair}")
                    continue
                
                if not os.path.exists(data_path):
                    self.logger.warning(f"Brak zapisanych danych dla {pair}")
                    continue
            
            # Załaduj model
                model = tf.keras.models.load_model(model_path)
            
            # Załaduj statystyki normalizacji
                with open(stats_path, 'r') as f:
                    norm_stats = json.load(f)
            
                # Załaduj dane
                with open(data_path, 'r') as f:
                    historical_data = json.load(f)
                
                # Sprawdź czy mamy wystarczająco danych
                if len(historical_data) < 10:
                    self.logger.warning(f"Za mało danych dla {pair} (potrzeba min. 10, mamy {len(historical_data)})")
                    continue
                    
                # Weź ostatnie 10 próbek
                recent_data = historical_data[-10:]
                prices = np.array([float(d['price']) for d in recent_data])
                volumes = np.array([float(d['volume_24h']) for d in recent_data])
                
                # Normalizacja
                normalized_prices = (prices - norm_stats['price_mean']) / norm_stats['price_std']
                normalized_volumes = (volumes - norm_stats['volume_mean']) / norm_stats['volume_std']
                
                # Przygotuj dane wejściowe
                X = np.column_stack((normalized_prices, normalized_volumes))
                X = X.reshape(1, 10, 2)
                
                # Zrób przewidywanie
                pred = model.predict(X, verbose=0)
                
                # Denormalizacja
                predicted_price = pred[0][0] * norm_stats['price_std'] + norm_stats['price_mean']
                current_price = prices[-1]
                price_change = ((predicted_price - current_price) / current_price) * 100
                
                self.logger.info(f"=== PRZEWIDYWANIE DLA {pair} ===")
                self.logger.info(f"Aktualna cena: {current_price:.2f} USD")
                self.logger.info(f"Przewidywana cena: {predicted_price:.2f} USD")
                self.logger.info(f"Przewidywana zmiana: {price_change:.2f}%")
                self.logger.info("===================")
            
                # Podejmij decyzję handlową
                decision = self.make_trading_decision(pair, current_price, predicted_price)
                if decision['action'] != 'hold':
                    self.execute_trade(pair, decision['action'], decision)
            
        except Exception as e:
            self.logger.error(f"Błąd podczas testowania przewidywań: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False
        
        return True

    def run(self):
        """Główna pętla bota"""
        self.logger.info("=== BOT URUCHOMIONY ===")
        self.is_running = True
        
        while self.is_running:
            try:
                # Sprawdź czy istnieje plik commands.txt
                if os.path.exists('commands.txt'):
                    self.logger.info("=== ZNALEZIONO PLIK COMMANDS.TXT ===")
                    with open('commands.txt', 'r') as f:
                        command = f.read().strip().lower()
                    os.remove('commands.txt')
                    self.logger.info(f"=== ODCZYTANO KOMENDĘ: {command} ===")
                    
                    if command == 'stop':
                        self.logger.info("=== OTRZYMANO POLECENIE ZATRZYMANIA ===")
                        self.stop_bot()
                        break
                    else:
                        self.logger.info(f"=== PRZEKAZUJĘ KOMENDĘ DO OBSŁUGI: {command} ===")
                        self.handle_command(command)
                        
            except Exception as e:
                self.logger.error(f"Błąd podczas obsługi pliku commands.txt: {e}")
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                # Usuń uszkodzony plik commands.txt
                if os.path.exists('commands.txt'):
                    try:
                        os.remove('commands.txt')
                    except:
                        pass
            time.sleep(1)

    def prepare_data_for_training(self):
        """Przygotowanie danych do trenowania modelu"""
        self.logger.info("=== PRZYGOTOWUJĘ DANE DO TRENOWANIA ===")
        
        # Słownik na dane dla każdej pary
        pair_data = {}
        
        for pair in self.trading_pairs:
            self.logger.info(f"Przetwarzam dane dla {pair}")
            data = self.market_data[pair]
            if len(data) < 100:
                self.logger.warning(f"Za mało danych dla {pair} (potrzeba min. 100, mamy {len(data)})")
                continue
            
            # Konwertuj dane na numpy array
            prices = np.array([float(d['price']) for d in data])
            volumes = np.array([float(d['volume_24h']) for d in data])
            
            # Normalizacja danych dla tej pary
            price_mean = np.mean(prices)
            price_std = np.std(prices)
            volume_mean = np.mean(volumes)
            volume_std = np.std(volumes)
            
            normalized_prices = (prices - price_mean) / price_std
            normalized_volumes = (volumes - volume_mean) / volume_std
            
            # Twórz sekwencje danych
            X = []
            y = []
            for i in range(len(normalized_prices) - 10):
                X.append(np.column_stack((normalized_prices[i:i+10], normalized_volumes[i:i+10])))
                next_price = normalized_prices[i+10]
                y.append(next_price)
            
            if X and y:
                pair_data[pair] = {
                    'X': np.array(X),
                    'y': np.array(y),
                    'stats': {
                        'price_mean': price_mean,
                        'price_std': price_std,
                        'volume_mean': volume_mean,
                        'volume_std': volume_std
                    }
                }
        
        if not pair_data:
            self.logger.error("=== BRAK DANYCH DO TRENOWANIA ===")
            return None
        
        return pair_data

    def calculate_rsi(self, prices: list, period: int = 14) -> float:
        """
        Oblicza wskaźnik RSI (Relative Strength Index).
        
        Args:
            prices: Lista cen zamknięcia
            period: Okres dla obliczenia RSI (domyślnie 14)
            
        Returns:
            float: Wartość RSI
        """
        try:
            if len(prices) < period + 1:
                return 50.0  # Wartość neutralna jeśli za mało danych
            
            # Oblicz zmiany cen
            deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
            
            # Rozdziel zmiany na wzrosty i spadki
            gains = [delta if delta > 0 else 0 for delta in deltas]
            losses = [-delta if delta < 0 else 0 for delta in deltas]
            
            # Oblicz średnie wzrosty i spadki
            avg_gain = sum(gains[-period:]) / period
            avg_loss = sum(losses[-period:]) / period
            
            if avg_loss == 0:
                return 100.0
            
            # Oblicz RSI
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            self.logger.error(f"Błąd podczas obliczania RSI: {str(e)}")
            return 50.0  # Wartość neutralna w przypadku błędu

    def analyze_volume(self, volumes: list, period: int = 20) -> dict:
        """
        Analizuje wolumen obrotu.
        
        Args:
            volumes: Lista wartości wolumenu
            period: Okres do analizy (domyślnie 20)
            
        Returns:
            dict: Słownik zawierający informacje o wolumenie
        """
        try:
            if len(volumes) < period:
                return {
                    'volume_ratio': 1.0,
                    'volume_trend': 'neutral',
                    'is_volume_increasing': False
                }
            
            # Oblicz średni wolumen
            avg_volume = sum(volumes[-period:]) / period
            
            # Oblicz stosunek aktualnego wolumenu do średniego
            current_volume = volumes[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Określ trend wolumenu
            recent_volumes = volumes[-5:]
            volume_trend = 'neutral'
            if all(recent_volumes[i] >= recent_volumes[i-1] for i in range(1, len(recent_volumes))):
                volume_trend = 'increasing'
            elif all(recent_volumes[i] <= recent_volumes[i-1] for i in range(1, len(recent_volumes))):
                volume_trend = 'decreasing'
            
            return {
                'volume_ratio': volume_ratio,
                'volume_trend': volume_trend,
                'is_volume_increasing': volume_trend == 'increasing'
            }
            
        except Exception as e:
            self.logger.error(f"Błąd podczas analizy wolumenu: {str(e)}")
            return {
                'volume_ratio': 1.0,
                'volume_trend': 'neutral',
                'is_volume_increasing': False
            }

    def calculate_volatility(self, prices: list, period: int = 20) -> dict:
        """
        Oblicza zmienność cen.
        
        Args:
            prices: Lista cen zamknięcia
            period: Okres do analizy (domyślnie 20)
            
        Returns:
            dict: Słownik zawierający informacje o zmienności
        """
        try:
            if len(prices) < period:
                return {
                    'volatility': 0.0,
                    'volatility_percentage': 0.0,
                    'is_high_volatility': False
                }
            
            # Oblicz zmiany procentowe
            returns = [(prices[i] - prices[i-1]) / prices[i-1] * 100 
                      for i in range(1, len(prices))]
            
            # Oblicz odchylenie standardowe
            mean_return = sum(returns[-period:]) / period
            squared_diff_sum = sum((r - mean_return) ** 2 for r in returns[-period:])
            volatility = (squared_diff_sum / period) ** 0.5
            
            # Oblicz zmienność jako procent
            volatility_percentage = volatility
            
            # Określ czy zmienność jest wysoka
            is_high_volatility = volatility_percentage > float(os.getenv('MAX_VOLATILITY', 5.0))
            
            return {
                'volatility': volatility,
                'volatility_percentage': volatility_percentage,
                'is_high_volatility': is_high_volatility
            }
            
        except Exception as e:
            self.logger.error(f"Błąd podczas obliczania zmienności: {str(e)}")
            return {
                'volatility': 0.0,
                'volatility_percentage': 0.0,
                'is_high_volatility': False
            }

if __name__ == "__main__":
    max_restart_attempts = 3
    restart_delay = 5
    restart_attempts = 0
    
    while restart_attempts < max_restart_attempts:
        try:
            bot = CryptoBotAI()
            bot.logger.info('=== INICJALIZACJA BOTA ===')

            # Start WebSocket w osobnym wątku
            ws_thread = threading.Thread(target=bot.connect_websocket, daemon=True)
            ws_thread.start()
            bot.logger.info('=== WĄTEK WEBSOCKET WYSTARTOWANY ===')

            # Główna pętla bota w głównym wątku
            bot.logger.info('=== START GŁÓWNEJ PĘTLI BOTA ===')
            bot.run()
                
        except Exception as e:
            restart_attempts += 1
            print(f"Błąd krytyczny: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            if restart_attempts < max_restart_attempts:
                print(f"Próba restartu {restart_attempts}/{max_restart_attempts} za {restart_delay} sekund...")
                time.sleep(restart_delay)
            else:
                print("Przekroczono maksymalną liczbę prób restartu.")
                break
        finally:
            if 'bot' in locals():
                bot.stop_bot()
            print("Bot został zatrzymany.") 