import time
import json
import os
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
from coinbase.rest import RESTClient
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from websocket_client import CoinbaseWebSocket
import threading
import websocket

# Konfiguracja loggera
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Dodaj handler do pliku
file_handler = logging.FileHandler('crypto_bot.log')
file_handler.setLevel(logging.INFO)

# Dodaj handler do konsoli
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Format logów
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Dodaj handlery do loggera
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Załaduj zmienne środowiskowe
load_dotenv('production.env')

# Konfiguracja API
with open('cdp_api_key.json', 'r') as f:
    api_keys = json.load(f)
    API_KEY = api_keys['api_key_id']
    API_SECRET = api_keys['api_key_secret']

# Parametry handlowe
TRADING_PAIRS = ['ETH-USDC', 'BTC-USDC']  # Tylko te dwie pary są obsługiwane
TRADE_VALUE_USDC = float(os.getenv('TRADE_VALUE_USDC', 50))  # Zwiększona wartość pojedynczej transakcji
MAX_TOTAL_EXPOSURE = float(os.getenv('MAX_TOTAL_EXPOSURE', 500))  # Zwiększona maksymalna ekspozycja całkowita
MAX_POSITION_SIZE = float(os.getenv('MAX_POSITION_SIZE', 100))  # Zwiększony maksymalny rozmiar pozycji
PRICE_THRESHOLD_BUY = float(os.getenv('PRICE_THRESHOLD_BUY', 3500))
PRICE_THRESHOLD_SELL = float(os.getenv('PRICE_THRESHOLD_SELL', 4000))
MIN_PROFIT_PERCENT = float(os.getenv('MIN_PROFIT_PERCENT', 0.3))
MAX_LOSS_PERCENT = float(os.getenv('MAX_LOSS_PERCENT', 0.5))

# Nowe parametry dla stop-loss i take-profit
STOP_LOSS_PERCENT = float(os.getenv('STOP_LOSS_PERCENT', 2.0))  # Stop-loss na -2%
TAKE_PROFIT_PERCENT = float(os.getenv('TAKE_PROFIT_PERCENT', 2.0))  # Take-profit na +2%
TRAILING_STOP_PERCENT = float(os.getenv('TRAILING_STOP_PERCENT', 1.0))  # Trailing stop 1%

# Parametry podwajania zysków
PROFIT_DOUBLING_DAYS = 3  # Co ile dni podwajamy zyski
INITIAL_PROFIT_TARGET = 1.0  # Początkowy cel zysku w procentach
MAX_PROFIT_TARGET = 10.0  # Maksymalny cel zysku w procentach
PROFIT_MULTIPLIER = 2.0  # Mnożnik do podwajania zysków

# Parametry zarządzania kapitałem
CAPITAL_ALLOCATION = {
    'ETH-USDC': 0.5,  # 50% kapitału na ETH
    'BTC-USDC': 0.5,  # 50% kapitału na BTC
    'RESERVE': 0.0    # Brak rezerwy
}

MIN_TRADE_SIZE_USDC = 30  # Zwiększona minimalna wielkość zlecenia
MAX_TRADES_PER_DAY = 10   # Zwiększona maksymalna liczba transakcji dziennie

# Parametry zarządzania zyskami
MIN_PROFIT_TARGET = 0.5  # Minimalny cel zysku w procentach
MAX_PROFIT_TARGET = 5.0  # Maksymalny cel zysku w procentach
VOLATILITY_MULTIPLIER = 2.0  # Mnożnik zmienności do obliczania celu zysku
TREND_STRENGTH_MULTIPLIER = 1.5  # Mnożnik siły trendu do obliczania celu zysku

# Minimalny rozmiar zlecenia dla każdej pary (możesz dostosować do wymagań giełdy)
MIN_ORDER_SIZE_USDC = {
    'ETH-USDC': 10.0,
    'BTC-USDC': 10.0
}

# Inicjalizacja API
client = RESTClient(api_key=API_KEY, api_secret=API_SECRET)

# Inicjalizacja WebSocket dla wszystkich par
ws = CoinbaseWebSocket(API_KEY, API_SECRET, TRADING_PAIRS)

# Globalne zmienne do przechowywania danych w czasie rzeczywistym
market_data = {}
for pair in TRADING_PAIRS:
    market_data[pair] = {
        'current_price': None,
        'order_book': {'bids': [], 'asks': []},
        'last_trade': None,
        'price_history': [],
        'last_buy_price': None,
        'last_sell_price': None,
        'trade_history': [],
        'highest_price': None
    }

def on_ws_message(data):
    """Obsługa wiadomości z WebSocket."""
    global current_price, order_book, last_trade, price_history
    
    try:
        if data['type'] == 'ticker':
            price = float(data.get('price', 0))
            if price > 0:
                current_price = price
                price_history.append({
                    'timestamp': datetime.utcnow(),
                    'price': price
                })
                # Zachowaj tylko ostatnie 1000 cen
                if len(price_history) > 1000:
                    price_history.pop(0)
                logging.info(f"Aktualna cena {SYMBOL}: {price}")
            
        elif data['type'] == 'snapshot':
            order_book['bids'] = data.get('bids', [])
            order_book['asks'] = data.get('asks', [])
            logging.info(f"Zaktualizowano książkę zleceń: {len(order_book['bids'])} bidów, {len(order_book['asks'])} asków")
            
        elif data['type'] == 'l2update':
            changes = data.get('changes', [])
            for change in changes:
                side, price, size = change
                if side == 'buy':
                    order_book['bids'] = [b for b in order_book['bids'] if b[0] != price]
                    if float(size) > 0:
                        order_book['bids'].append([price, size])
                else:
                    order_book['asks'] = [a for a in order_book['asks'] if a[0] != price]
                    if float(size) > 0:
                        order_book['asks'].append([price, size])
            
            # Sortowanie książki zleceń
            order_book['bids'].sort(key=lambda x: float(x[0]), reverse=True)
            order_book['asks'].sort(key=lambda x: float(x[0]))
            
    except Exception as e:
        logging.error(f"Błąd podczas przetwarzania wiadomości WebSocket: {e}")

def calculate_market_depth():
    """Oblicza głębokość rynku na podstawie książki zleceń."""
    try:
        total_bids = sum(float(bid[1]) for bid in order_book['bids'][:10])  # Top 10 bidów
        total_asks = sum(float(ask[1]) for ask in order_book['asks'][:10])  # Top 10 asków
        
        return {
            'bids_volume': total_bids,
            'asks_volume': total_asks,
            'ratio': total_bids / total_asks if total_asks > 0 else 0
        }
    except Exception as e:
        logging.error(f"Błąd podczas obliczania głębokości rynku: {e}")
        return None

def calculate_volatility(historical_data):
    """Oblicza zmienność ceny na podstawie historii cen."""
    try:
        if historical_data is None or len(historical_data) < 2:
            return 0
            
        # Użyj kolumny 'price' z DataFrame
        prices = historical_data['price'].values
        returns = np.diff(prices) / prices[:-1]
        return np.std(returns) * 100  # Zmienność w procentach
    except Exception as e:
        logging.error(f"Błąd podczas obliczania zmienności: {e}")
        return 0

def calculate_rsi(prices, period=14):
    """Oblicza wskaźnik RSI na podstawie historii cen."""
    try:
        if len(prices) < period:
            logging.warning(f"Za mało danych do obliczenia RSI (potrzeba {period}, jest {len(prices)})")
            return 50
            
        # Użyj bezpośrednio kolumny 'price' z DataFrame
        price_series = prices['price']
        
        # Oblicz zmiany cen
        delta = price_series.diff()
        
        # Oblicz zyski i straty
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # Oblicz RS i RSI
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
    except Exception as e:
        logging.error(f"Błąd podczas obliczania RSI: {e}")
        return 50

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Oblicza wskaźnik MACD."""
    try:
        if len(prices) < slow:
            logging.warning(f"Za mało danych do obliczenia MACD (potrzeba {slow}, jest {len(prices)})")
            return None, None, None
        
        # Użyj bezpośrednio kolumny 'price' z DataFrame
        price_series = prices['price']
        
        # Obliczanie EMA
        ema_fast = price_series.ewm(span=fast, adjust=False).mean()
        ema_slow = price_series.ewm(span=slow, adjust=False).mean()
        
        # Obliczanie MACD
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1]
    except Exception as e:
        logging.error(f"Błąd podczas obliczania MACD: {e}")
        return None, None, None

def calculate_moving_averages(prices, short_period=20, long_period=50):
    """Oblicza średnie kroczące."""
    try:
        if len(prices) < long_period:
            logging.warning(f"Za mało danych do obliczenia średnich kroczących (potrzeba {long_period}, jest {len(prices)})")
            return None, None
        
        # Użyj bezpośrednio kolumny 'price' z DataFrame
        price_series = prices['price']
        
        # Oblicz średnie kroczące
        ma_short = price_series.rolling(window=short_period).mean().iloc[-1]
        ma_long = price_series.rolling(window=long_period).mean().iloc[-1]
        
        return ma_short, ma_long
    except Exception as e:
        logging.error(f"Błąd podczas obliczania średnich kroczących: {e}")
        return None, None

def get_usdc_balance():
    """Pobierz saldo USDC."""
    try:
        accounts = client.get_accounts().accounts
        for account in accounts:
            if account.currency == 'USDC':
                balance = account.available_balance
                if balance:
                    if isinstance(balance, dict):
                        return float(balance.get('value', 0))
                    else:
                        return float(balance.value)
        return 0
    except Exception as e:
        logging.error(f"Błąd podczas pobierania salda USDC: {e}")
        return 0

def get_crypto_balance(crypto_symbol):
    """Pobierz saldo danej kryptowaluty."""
    try:
        accounts = client.get_accounts().accounts
        for account in accounts:
            if account.currency == crypto_symbol:
                balance = account.available_balance
                if balance:
                    if isinstance(balance, dict):
                        return float(balance.get('value', 0))
                    else:
                        return float(balance.value)
        return 0
    except Exception as e:
        logging.error(f"Błąd podczas pobierania salda {crypto_symbol}: {e}")
        return 0

def should_trade(current_price, historical_data):
    """Decyduje, czy należy wykonać transakcję na podstawie analizy technicznej."""
    try:
        if historical_data is None or len(historical_data) < 24:
            logging.info("Za mało danych historycznych do analizy")
            return False
            
        # Oblicz wskaźniki techniczne
        rsi = calculate_rsi(historical_data)
        macd, signal, hist = calculate_macd(historical_data)
        ma_short, ma_long = calculate_moving_averages(historical_data)
        
        if macd is None or ma_short is None:
            logging.info("Nie można obliczyć wskaźników technicznych")
            return False
            
        # Loguj wskaźniki
        logging.info(f"RSI: {rsi:.2f}")
        logging.info(f"MACD: {macd:.2f}, Signal: {signal:.2f}, Hist: {hist:.2f}")
        logging.info(f"MA20: {ma_short:.2f}, MA50: {ma_long:.2f}")
        
        # Oblicz zmienność ceny
        price_volatility = calculate_volatility(historical_data)
        logging.info(f"Zmienność ceny: {price_volatility:.2f}%")
        
        # Bardzo selektywne warunki kupna - tylko silne sygnały
        if (current_price > ma_short and rsi < 45 and macd > signal and hist > 0) or \
           (current_price > ma_long and rsi < 40 and price_volatility > 0.5) or \
           (current_price > ma_short and hist > 0 and rsi < 50 and price_volatility > 0.3):
            logging.info("SILNY SYGNAŁ KUPNA - Wykryto potencjalny wzrost")
            return True
            
        # Bardzo selektywne warunki sprzedaży - tylko przy pewnych sygnałach
        if (current_price < ma_short and rsi > 55) or \
           (current_price < ma_long and rsi > 50) or \
           (current_price < ma_short and hist < 0 and rsi > 45 and price_volatility > 0.8):
            logging.info("SILNY SYGNAŁ SPRZEDAŻY - Realizacja zysku lub ograniczenie straty")
            return True
            
        return False
        
    except Exception as e:
        logging.error(f"Błąd podczas analizy warunków handlowych: {e}")
        return False

def init_ai_model():
    """Inicjalizacja modelu AI do przewidywania cen."""
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(60, 1)),
        tf.keras.layers.LSTM(50, return_sequences=False),
        tf.keras.layers.Dense(25),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def get_historical_data(product_id):
    """Pobierz historyczne dane cenowe dla danej pary."""
    try:
        # Pobierz świeczki (candles) dla danego produktu
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=3)  # Pobierz dane z ostatnich 3 dni
        
        # Konwertuj daty na timestamp w sekundach
        start_timestamp = int(start_time.timestamp())
        end_timestamp = int(end_time.timestamp())
        
        # Użyj prawidłowej metody API z timestampami w sekundach
        response = client.get_candles(
            product_id=product_id,
            start=start_timestamp,
            end=end_timestamp,
            granularity="ONE_HOUR"  # Prawidłowa wartość dla Coinbase Advanced API
        )
        
        if not response or not hasattr(response, 'candles'):
            logging.warning(f"Brak danych historycznych dla {product_id}")
            return None
            
        # Loguj strukturę pierwszej świeczki
        if response.candles:
            first_candle = response.candles[0]
            logging.info(f"Struktura pierwszej świeczki: {dir(first_candle)}")
            logging.info(f"Wartości pierwszej świeczki: {first_candle.__dict__}")
            
        # Konwertuj dane do DataFrame
        data = []
        for candle in response.candles:
            try:
                data.append({
                    'timestamp': datetime.fromtimestamp(int(candle.start)),
                    'price': float(candle.close),
                    'size': float(candle.volume)
                })
            except (AttributeError, ValueError) as e:
                logging.warning(f"Pominięto nieprawidłową świeczkę dla {product_id}: {e}")
                continue
        
        if not data:
            logging.warning(f"Brak prawidłowych danych historycznych dla {product_id}")
            return None
            
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        logging.info(f"Pobrano {len(df)} świeczek dla {product_id}")
        logging.info(f"Przykładowe dane: {df.head().to_dict()}")
        return df
        
    except Exception as e:
        logging.error(f"Błąd podczas pobierania historycznych danych dla {product_id}: {e}")
        return None

def prepare_data_for_ai(historical_data):
    """Przygotuj dane dla modelu AI."""
    if not historical_data:
        return None
    
    try:
        # Normalizacja danych
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(historical_data['price'].values.reshape(-1, 1))
        
        # Przygotowanie sekwencji
        X, y = [], []
        for i in range(60, len(scaled_data)):
            X.append(scaled_data[i-60:i, 0])
            y.append(scaled_data[i, 0])
        
        return np.array(X), np.array(y), scaler
    except Exception as e:
        logging.error(f"Błąd podczas przygotowywania danych dla AI: {e}")
        return None

def predict_price(model, X, scaler):
    """Przewiduj przyszłą cenę."""
    try:
        prediction = model.predict(X)
        return scaler.inverse_transform(prediction)[0][0]
    except Exception as e:
        logging.error(f"Błąd podczas przewidywania ceny: {e}")
        return None

def analyze_market_trend(historical_data):
    """Analizuj trend rynkowy."""
    if not historical_data:
        return "neutral"
    
    try:
        # Oblicz średnią kroczącą
        historical_data['MA20'] = historical_data['price'].rolling(window=20).mean()
        historical_data['MA50'] = historical_data['price'].rolling(window=50).mean()
        
        # Oblicz RSI (Relative Strength Index)
        delta = historical_data['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        historical_data['RSI'] = 100 - (100 / (1 + rs))
        
        current_price = historical_data['price'].iloc[-1]
        ma20 = historical_data['MA20'].iloc[-1]
        ma50 = historical_data['MA50'].iloc[-1]
        rsi = historical_data['RSI'].iloc[-1]
        
        # Analiza trendu na podstawie wielu wskaźników
        if current_price > ma20 and ma20 > ma50 and rsi > 50:
            return "bullish"
        elif current_price < ma20 and ma20 < ma50 and rsi < 50:
            return "bearish"
        else:
            return "neutral"
    except Exception as e:
        logging.error(f"Błąd podczas analizy trendu: {e}")
        return "neutral"

def suggest_thresholds(current_price, trend, predicted_price=None):
    """Sugeruj progi kupna/sprzedaży na podstawie analizy AI."""
    try:
        if predicted_price:
            # Jeśli mamy przewidywanie ceny, użyj go do ustawienia progów
            price_range = abs(predicted_price - current_price)
            if trend == "bullish":
                return {
                    'buy': current_price * 0.95,  # 5% poniżej aktualnej ceny
                    'sell': predicted_price * 1.05  # 5% powyżej przewidywanej ceny
                }
            elif trend == "bearish":
                return {
                    'buy': predicted_price * 0.95,  # 5% poniżej przewidywanej ceny
                    'sell': current_price * 1.05  # 5% powyżej aktualnej ceny
                }
            else:
                return {
                    'buy': min(current_price, predicted_price) * 0.95,
                    'sell': max(current_price, predicted_price) * 1.05
                }
        else:
            # Jeśli nie mamy przewidywania, użyj standardowych progów
            if trend == "bullish":
                return {
                    'buy': current_price * 0.95,  # 5% poniżej aktualnej ceny
                    'sell': current_price * 1.15  # 15% powyżej aktualnej ceny
                }
            elif trend == "bearish":
                return {
                    'buy': current_price * 0.85,  # 15% poniżej aktualnej ceny
                    'sell': current_price * 1.05  # 5% powyżej aktualnej ceny
                }
            else:
                return {
                    'buy': current_price * 0.90,  # 10% poniżej aktualnej ceny
                    'sell': current_price * 1.10  # 10% powyżej aktualnej ceny
                }
    except Exception as e:
        logging.error(f"Błąd podczas sugerowania progów: {e}")
        return {
            'buy': current_price * 0.90,
            'sell': current_price * 1.10
        }

def get_account_balance():
    """Pobierz stan konta."""
    try:
        accounts = client.get_accounts().accounts
        usdc_balance = 0
        eth_balance = 0
        for account in accounts:
            if account.currency == 'USDC':
                balance = account.available_balance
                if balance:
                    if isinstance(balance, dict):
                        usdc_balance = float(balance.get('value', 0))
                    else:
                        usdc_balance = float(balance.value)
            elif account.currency == 'ETH':
                balance = account.available_balance
                if balance:
                    if isinstance(balance, dict):
                        eth_balance = float(balance.get('value', 0))
                    else:
                        eth_balance = float(balance.value)
        return usdc_balance, eth_balance
    except Exception as e:
        logging.error(f"Błąd podczas pobierania stanu konta: {e}")
        raise

def get_current_price(product_id):
    """Pobierz aktualną cenę dla danej pary."""
    try:
        # Sprawdź czy para jest na liście dostępnych par
        if product_id not in TRADING_PAIRS:
            logging.warning(f"Para {product_id} nie jest na liście dostępnych par handlowych")
            return None

        product = client.get_product(product_id)
        return float(product.price)
    except Exception as e:
        logging.error(f"Błąd podczas pobierania aktualnej ceny dla {product_id}: {e}")
        return None

def check_total_exposure():
    """Sprawdza całkowitą ekspozycję na rynku."""
    try:
        total_exposure = 0
        balances = get_all_balances()
        
        for currency, data in balances.items():
            if currency != 'USDC':
                try:
                    # Sprawdź czy para istnieje przed próbą pobrania ceny
                    pair = f"{currency}-USDC"
                    if pair not in TRADING_PAIRS:
                        logging.debug(f"Para {pair} nie jest obsługiwana - pomijam")
                        continue
                        
                    current_price = get_current_price(pair)
                    if current_price is None:
                        logging.debug(f"Nie można pobrać ceny dla {pair} - pomijam")
                        continue
                        
                    position_value = float(data['total']) * current_price
                    total_exposure += position_value
                    logging.debug(f"Ekspozycja dla {pair}: {position_value:.2f} USDC")
                except Exception as e:
                    logging.debug(f"Błąd przy obliczaniu ekspozycji dla {currency}: {e}")
                    continue
                    
        logging.info(f"Całkowita ekspozycja: {total_exposure:.2f} USDC")
        return total_exposure
    except Exception as e:
        logging.error(f"Błąd podczas sprawdzania ekspozycji: {e}")
        return 0

def can_open_new_position():
    """Sprawdza czy można otworzyć nową pozycję."""
    try:
        current_exposure = check_total_exposure()
        if current_exposure >= MAX_TOTAL_EXPOSURE:
            logging.info(f"Osiągnięto maksymalną ekspozycję: {current_exposure} USDC")
            return False
        return True
    except Exception as e:
        logging.error(f"Błąd podczas sprawdzania możliwości otwarcia pozycji: {e}")
        return False

def calculate_profit_target(product_id, historical_data):
    """Oblicza dynamiczny cel zysku na podstawie analizy rynku."""
    try:
        if historical_data is None or len(historical_data) < 24:
            return MIN_PROFIT_TARGET
            
        # Oblicz zmienność ceny
        volatility = calculate_volatility(historical_data)
        
        # Oblicz siłę trendu
        rsi = calculate_rsi(historical_data)
        macd, signal, hist = calculate_macd(historical_data)
        ma_short, ma_long = calculate_moving_averages(historical_data)
        
        if macd is None or ma_short is None:
            return MIN_PROFIT_TARGET
            
        # Określ siłę trendu
        trend_strength = 1.0
        if current_price > ma_short and ma_short > ma_long:
            trend_strength = TREND_STRENGTH_MULTIPLIER
        elif current_price < ma_short and ma_short < ma_long:
            trend_strength = 0.5
            
        # Oblicz cel zysku na podstawie zmienności i trendu
        profit_target = volatility * VOLATILITY_MULTIPLIER * trend_strength
        
        # Ogranicz cel zysku do rozsądnych wartości
        profit_target = max(MIN_PROFIT_TARGET, min(profit_target, MAX_PROFIT_TARGET))
        
        logging.info(f"Obliczony cel zysku dla {product_id}: {profit_target:.2f}% (zmienność: {volatility:.2f}%, siła trendu: {trend_strength:.2f})")
        return profit_target
        
    except Exception as e:
        logging.error(f"Błąd podczas obliczania celu zysku: {e}")
        return MIN_PROFIT_TARGET

def should_take_profit(product_id, current_price, historical_data):
    """Sprawdza czy należy zrealizować zysk na podstawie dynamicznej analizy rynku."""
    try:
        last_buy_price = market_data[product_id].get('last_buy_price')
        if not last_buy_price:
            return False
            
        # Oblicz procentową zmianę ceny
        price_change_percent = ((current_price - last_buy_price) / last_buy_price) * 100
        
        # Pobierz dynamiczny cel zysku
        profit_target = calculate_profit_target(product_id, historical_data)
        
        # Sprawdź czy osiągnięto cel zysku
        if price_change_percent >= profit_target:
            logging.info(f"Osiągnięto cel zysku {profit_target:.2f}% dla {product_id} (zmiana: {price_change_percent:.2f}%)")
            return True
            
        # Sprawdź czy trend się odwraca
        rsi = calculate_rsi(historical_data)
        macd, signal, hist = calculate_macd(historical_data)
        
        if price_change_percent > 0 and rsi > 70 and hist < 0:
            logging.info(f"Trend się odwraca - realizacja zysku {price_change_percent:.2f}% dla {product_id}")
            return True
            
        return False
        
    except Exception as e:
        logging.error(f"Błąd podczas sprawdzania realizacji zysku: {e}")
        return False

def check_stop_loss_take_profit(product_id, current_price):
    """Sprawdza warunki stop-loss i take-profit dla danej pozycji."""
    try:
        if product_id not in market_data:
            return False
            
        last_buy_price = market_data[product_id].get('last_buy_price')
        if not last_buy_price:
            return False
            
        # Pobierz dane historyczne
        historical_data = market_data[product_id].get('price_history')
        
        # Sprawdź stop-loss
        price_change_percent = ((current_price - last_buy_price) / last_buy_price) * 100
        if price_change_percent <= -STOP_LOSS_PERCENT:
            logging.info(f"STOP-LOSS dla {product_id} przy cenie {current_price} (zmiana: {price_change_percent:.2f}%)")
            return True
            
        # Sprawdź dynamiczny take-profit
        if should_take_profit(product_id, current_price, historical_data):
            return True
            
        # Sprawdź trailing stop
        highest_price = market_data[product_id].get('highest_price', last_buy_price)
        if current_price > highest_price:
            market_data[product_id]['highest_price'] = current_price
        elif current_price < highest_price * (1 - TRAILING_STOP_PERCENT/100):
            logging.info(f"TRAILING-STOP dla {product_id} przy cenie {current_price} (najwyższa: {highest_price})")
            return True
            
        return False
    except Exception as e:
        logging.error(f"Błąd podczas sprawdzania stop-loss/take-profit: {e}")
        return False

def get_available_capital_for_pair(product_id):
    """Oblicza dostępny kapitał dla danej pary handlowej."""
    try:
        total_capital = get_usdc_balance()
        allocation = CAPITAL_ALLOCATION.get(product_id, 0)
        return total_capital * allocation
    except Exception as e:
        logging.error(f"Błąd podczas obliczania dostępnego kapitału: {e}")
        return 0

def can_trade_today(product_id):
    """Sprawdza czy można wykonać kolejną transakcję dzisiaj."""
    try:
        today = datetime.utcnow().date()
        trades_today = sum(1 for trade in market_data[product_id]['trade_history'] 
                         if trade['timestamp'].date() == today)
        return trades_today < MAX_TRADES_PER_DAY
    except Exception as e:
        logging.error(f"Błąd podczas sprawdzania limitów dziennych: {e}")
        return False

def calculate_trade_size(product_id, current_price):
    """Oblicza optymalną wielkość zlecenia."""
    try:
        available_capital = get_available_capital_for_pair(product_id)
        min_size = MIN_ORDER_SIZE_USDC.get(product_id, 10.0)
        if available_capital < min_size:
            logging.info(f"Za mało kapitału na zlecenie dla {product_id} (min: {min_size} USDC)")
            return 0
        # Oblicz wielkość zlecenia jako 20% dostępnego kapitału, ale nie mniej niż min_size i nie więcej niż MAX_POSITION_SIZE
        trade_size = max(min_size, min(available_capital * 0.2, MAX_POSITION_SIZE))
        amount = trade_size / current_price
        logging.info(f"Obliczona wielkość zlecenia dla {product_id}: {trade_size} USDC ({amount} {product_id.split('-')[0]})")
        return trade_size
    except Exception as e:
        logging.error(f"Błąd podczas obliczania wielkości zlecenia: {e}")
        return 0

def place_buy_order(product_id, amount, price):
    """Złóż zlecenie kupna dla danej pary."""
    try:
        if not can_open_new_position():
            logging.info("Nie można otworzyć nowej pozycji - przekroczono maksymalną ekspozycję")
            return None
        if not can_trade_today(product_id):
            logging.info(f"Osiągnięto dzienny limit transakcji dla {product_id}")
            return None
        trade_size = calculate_trade_size(product_id, price)
        if trade_size == 0:
            logging.info(f"Za mało kapitału na zlecenie dla {product_id}")
            return None
        crypto_amount = trade_size / price
        logging.info(f"Próba złożenia zlecenia kupna dla {product_id}: {crypto_amount} {product_id.split('-')[0]} (wartość: {trade_size} USDC)")
        order = client.create_order(
            client_order_id=str(int(time.time() * 1000)),
            product_id=product_id,
            side="BUY",
            order_configuration={
                "market_market_ioc": {
                    "quote_size": str(trade_size)
                }
            }
        )
        market_data[product_id]['last_buy_price'] = price
        market_data[product_id]['highest_price'] = price
        market_data[product_id]['trade_history'].append({
            'type': 'buy',
            'price': price,
            'amount': crypto_amount,
            'timestamp': datetime.utcnow()
        })
        logging.info(f"Złożono zlecenie kupna dla {product_id}: {order}")
        return order
    except Exception as e:
        logging.error(f"Błąd podczas składania zlecenia kupna dla {product_id}: {e}")
        raise

def place_sell_order(product_id, amount, price):
    """Złóż zlecenie sprzedaży dla danej pary."""
    try:
        order = client.create_order(
            client_order_id=str(int(time.time() * 1000)),
            product_id=product_id,
            side="SELL",
            order_configuration={
                "market_market_ioc": {
                    "base_size": str(amount)
                }
            }
        )
        market_data[product_id]['last_sell_price'] = price
        market_data[product_id]['trade_history'].append({
            'type': 'sell',
            'price': price,
            'amount': amount,
            'timestamp': datetime.utcnow()
        })
        logging.info(f"Złożono zlecenie sprzedaży dla {product_id}: {order}")
        return order
    except Exception as e:
        logging.error(f"Błąd podczas składania zlecenia sprzedaży dla {product_id}: {e}")
        raise

def get_all_balances():
    """Pobierz wszystkie salda konta."""
    try:
        accounts = client.get_accounts().accounts
        balances = {}
        for account in accounts:
            if hasattr(account, 'available_balance') and account.available_balance:
                currency = account.currency
                # Obsługa różnych formatów danych
                if isinstance(account.available_balance, dict):
                    available = float(account.available_balance.get('value', 0))
                else:
                    available = float(account.available_balance.value)
                
                if hasattr(account, 'total_balance'):
                    if isinstance(account.total_balance, dict):
                        total = float(account.total_balance.get('value', 0))
                    else:
                        total = float(account.total_balance.value)
                else:
                    total = available
                
                if hasattr(account, 'hold'):
                    if isinstance(account.hold, dict):
                        hold = float(account.hold.get('value', 0))
                    else:
                        hold = float(account.hold.value)
                else:
                    hold = 0

                # Sprawdź czy konto ma informacje o stakowaniu
                staked = 0
                if hasattr(account, 'staked_balance'):
                    if isinstance(account.staked_balance, dict):
                        staked = float(account.staked_balance.get('value', 0))
                    else:
                        staked = float(account.staked_balance.value)
                
                # Jeśli nie znaleziono bezpośrednio staked_balance, spróbuj obliczyć z różnicy
                if staked == 0 and total > (available + hold):
                    staked = total - available - hold
                
                balances[currency] = {
                    'available': available,
                    'total': total,
                    'hold': hold,
                    'staked': staked,
                    'currency': currency,
                    'type': getattr(account, 'type', 'unknown'),
                    'active': getattr(account, 'active', True)
                }
        return balances
    except Exception as e:
        logging.error(f"Błąd podczas pobierania sald: {e}")
        raise

def print_detailed_balances():
    """Wyświetla szczegółowe informacje o saldach konta."""
    try:
        print("\n=== SPRAWDZANIE SALD KONTA ===")
        print("Pobieranie danych...")
        
        balances = get_all_balances()
        if not balances:
            print("Nie udało się pobrać sald - brak danych")
            return
            
        print("\n{:<10} {:<15} {:<15} {:<15} {:<15}".format(
            "Waluta", "Dostępne", "Całkowite", "Zablokowane", "Stakowane"
        ))
        print("-" * 75)
        
        total_value_usdc = 0
        
        for currency, data in balances.items():
            try:
                available = float(data['available'])
                total = float(data['total'])
                hold = float(data['hold'])
                staked = float(data['staked'])
                
                # Jeśli to nie USDC, sprawdź czy para handlowa istnieje
                if currency != 'USDC':
                    pair = f"{currency}-USDC"
                    if pair in TRADING_PAIRS:
                        try:
                            current_price = get_current_price(pair)
                            if current_price is not None:
                                value_usdc = total * current_price
                                total_value_usdc += value_usdc
                            else:
                                value_usdc = 0
                        except:
                            value_usdc = 0
                    else:
                        value_usdc = 0
                else:
                    value_usdc = total
                    total_value_usdc += value_usdc
                
                print("{:<10} {:<15.8f} {:<15.8f} {:<15.8f} {:<15.8f}".format(
                    currency,
                    available,
                    total,
                    hold,
                    staked
                ))
                
                if currency != 'USDC' and value_usdc > 0:
                    print(f"Wartość w USDC: {value_usdc:.2f}")
                    
            except Exception as e:
                print(f"Błąd przy przetwarzaniu salda {currency}: {e}")
                continue
        
        print("\n=== PODSUMOWANIE ===")
        print(f"Całkowita wartość w USDC: {total_value_usdc:.2f}")
        print("=========================")
        
    except Exception as e:
        print(f"Błąd podczas pobierania sald: {str(e)}")
        print(f"Typ błędu: {type(e).__name__}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

def get_transaction_history(product_id):
    """Pobiera historię transakcji dla danej pary."""
    try:
        print(f"\n=== HISTORIA TRANSAKCJI DLA {product_id} ===")
        print("Pobieranie danych...")
        
        fills = client.get_fills(product_id=product_id, limit=100)
        if not fills or not hasattr(fills, 'fills'):
            print("Brak historii transakcji")
            return
            
        if not fills.fills:
            print("Brak transakcji dla tej pary")
            return
            
        print(f"\nZnaleziono {len(fills.fills)} transakcji:")
        print("\n{:<20} {:<10} {:<15} {:<15} {:<15} {:<10}".format(
            "Data", "Typ", "Cena", "Ilość", "Wartość", "Opłata"
        ))
        print("-" * 90)
        
        for fill in fills.fills:
            try:
                data = getattr(fill, 'trade_time', '-')
                typ = 'KUPNO' if getattr(fill, 'side', '-') == 'BUY' else 'SPRZEDAŻ'
                cena = float(getattr(fill, 'price', 0))
                ilosc = float(getattr(fill, 'size', 0))
                wartosc = cena * ilosc
                oplata = getattr(fill, 'fee', '0')
                
                print("{:<20} {:<10} {:<15.8f} {:<15.8f} {:<15.2f} {:<10}".format(
                    str(data),
                    typ,
                    cena,
                    ilosc,
                    wartosc,
                    str(oplata)
                ))
            except Exception as e:
                print(f"Błąd przy przetwarzaniu transakcji: {e}")
                continue
                
        print("\n=========================")
    except Exception as e:
        print(f"Błąd podczas pobierania historii transakcji dla {product_id}: {e}")

def get_authenticated_client():
    """Tworzy i zwraca uwierzytelnionego klienta Coinbase."""
    try:
        client = RESTClient()
        return client
    except Exception as e:
        logger.error(f"Błąd podczas tworzenia klienta: {str(e)}")
        return None

def main():
    """Główna funkcja bota"""
    try:
        # Inicjalizacja klienta
        client = get_authenticated_client()
        if not client:
            logger.error("Nie udało się zainicjalizować klienta")
            return

        # Inicjalizacja WebSocket
        ws.connect()
        time.sleep(2)  # Czekamy na połączenie

        # Subskrybujemy się do kanałów tylko raz
        for pair in TRADING_PAIRS:
            try:
                ws.subscribe_to_ticker(pair)
                time.sleep(0.5)  # Dodajemy opóźnienie między subskrypcjami
                ws.subscribe_to_level2(pair)
                time.sleep(0.5)
                ws.subscribe_to_market_trades(pair)
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"Błąd podczas subskrybowania do kanałów dla {pair}: {str(e)}")
                continue

        # Główna pętla
        while True:
            try:
                # Sprawdzamy stan połączenia
                if not ws.is_connected():
                    logger.warning("Utracono połączenie WebSocket, próba ponownego połączenia...")
                    ws.connect()
                    time.sleep(2)  # Czekamy na połączenie
                    continue

                # Sprawdzamy salda
                accounts = client.get_accounts()
                usdc_balance = 0.0
                btc_balance = 0.0
                eth_balance = 0.0

                for account in accounts:
                    if account.currency == 'USDC':
                        usdc_balance = float(account.available_balance.value)
                    elif account.currency == 'BTC':
                        btc_balance = float(account.available_balance.value)
                    elif account.currency == 'ETH':
                        eth_balance = float(account.available_balance.value)

                logger.info(f"Saldo USDC: {usdc_balance}")
                logger.info(f"Saldo BTC: {btc_balance}")
                logger.info(f"Saldo ETH: {eth_balance}")

                # Sprawdzamy każdą parę
                for pair in TRADING_PAIRS:
                    try:
                        # Pobieramy świeczki
                        end_time = datetime.now()
                        start_time = end_time - timedelta(days=3)
                        candles = client.get_product_candles(
                            product_id=pair,
                            start=start_time,
                            end=end_time,
                            granularity='ONE_HOUR'
                        )

                        if not candles:
                            logger.warning(f"Brak danych świeczek dla {pair}")
                            continue

                        # Konwertujemy dane
                        df = pd.DataFrame([{
                            'timestamp': candle.start,
                            'price': float(candle.close),
                            'size': float(candle.volume)
                        } for candle in candles])

                        # Obliczamy wskaźniki
                        df['rsi'] = calculate_rsi(df['price'])
                        macd, signal, hist = calculate_macd(df['price'])
                        df['macd'] = macd
                        df['signal'] = signal
                        df['hist'] = hist
                        df['ma20'] = calculate_ma(df['price'], 20)
                        df['ma50'] = calculate_ma(df['price'], 50)

                        # Pobieramy aktualną cenę
                        ticker = client.get_product_ticker(product_id=pair)
                        if not ticker:
                            logger.warning(f"Nie udało się pobrać aktualnej ceny dla {pair}")
                            continue

                        current_price = float(ticker.price)
                        logger.info(f"Aktualna cena {pair}: {current_price}")

                        # Sprawdzamy sygnały
                        last_row = df.iloc[-1]
                        logger.info(f"RSI: {last_row['rsi']:.2f}")
                        logger.info(f"MACD: {last_row['macd']:.2f}, Signal: {last_row['signal']:.2f}, Hist: {last_row['hist']:.2f}")
                        logger.info(f"MA20: {last_row['ma20']:.2f}, MA50: {last_row['ma50']:.2f}")

                        # Obliczamy zmienność ceny
                        price_change = (current_price - df['price'].iloc[-2]) / df['price'].iloc[-2] * 100
                        logger.info(f"Zmienność ceny: {price_change:.2f}%")

                        # Sprawdzamy warunki handlowe
                        if pair == 'BTC-USDC':
                            if btc_balance > 0:
                                if (last_row['rsi'] > 70 or 
                                    (last_row['macd'] < last_row['signal'] and last_row['hist'] < 0) or
                                    current_price < last_row['ma20']):
                                    logger.info("SILNY SYGNAŁ SPRZEDAŻY - Realizacja zysku lub ograniczenie straty")
                                    place_sell_order(client, pair, btc_balance)
                        else:  # ETH-USDC
                            if usdc_balance >= MIN_TRADE_SIZE_USDC:
                                if (last_row['rsi'] < 30 or 
                                    (last_row['macd'] > last_row['signal'] and last_row['hist'] > 0) or
                                    current_price > last_row['ma20']):
                                    logger.info("SILNY SYGNAŁ KUPNA - Wejście w pozycję")
                                    place_buy_order(client, pair, usdc_balance)

                    except Exception as e:
                        logger.error(f"Błąd podczas przetwarzania pary {pair}: {str(e)}")
                        continue

                time.sleep(60)  # Czekamy minutę przed następnym sprawdzeniem

            except Exception as e:
                logger.error(f"Błąd w głównej pętli: {str(e)}")
                time.sleep(60)  # W przypadku błędu czekamy minutę przed ponowną próbą

    except KeyboardInterrupt:
        logger.info("Zatrzymywanie bota...")
    except Exception as e:
        logger.error(f"Nieoczekiwany błąd: {str(e)}")
    finally:
        if 'ws' in locals():
            ws.close()

if __name__ == "__main__":
    print_detailed_balances()
    print("\nSprawdzanie historii transakcji dla wszystkich par...")
    for pair in TRADING_PAIRS:
        get_transaction_history(pair)
    main()




