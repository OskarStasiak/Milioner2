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
TRADING_PAIRS = ['ETH-USDC', 'BTC-USDC', 'ETH-USD', 'BTC-USD']  # Wszystkie obsługiwane pary
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
    'ETH-USDC': 0.25,  # 25% kapitału na ETH-USDC
    'BTC-USDC': 0.25,  # 25% kapitału na BTC-USDC
    'ETH-USD': 0.25,   # 25% kapitału na ETH-USD
    'BTC-USD': 0.25,   # 25% kapitału na BTC-USD
    'RESERVE': 0.0     # Brak rezerwy
}

MIN_TRADE_SIZE_USDC = 10  # Zmniejszona minimalna wielkość zlecenia
MAX_TRADES_PER_DAY = 20   # Zwiększona maksymalna liczba transakcji dziennie

# Parametry zarządzania zyskami
MIN_PROFIT_TARGET = 0.5  # Minimalny cel zysku w procentach
MAX_PROFIT_TARGET = 5.0  # Maksymalny cel zysku w procentach
VOLATILITY_MULTIPLIER = 2.0  # Mnożnik zmienności do obliczania celu zysku
TREND_STRENGTH_MULTIPLIER = 1.5  # Mnożnik siły trendu do obliczania celu zysku

# Minimalny rozmiar zlecenia dla każdej pary (możesz dostosować do wymagań giełdy)
MIN_ORDER_SIZE_USDC = {
    'ETH-USDC': 10.0,
    'BTC-USDC': 10.0,
    'ETH-USD': 10.0,
    'BTC-USD': 10.0
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
        'trade_history': [],  # Inicjalizacja pustej historii transakcji
        'highest_price': None
    }
    logging.info(f"Inicjalizacja danych dla {pair} - historia transakcji wyczyszczona")

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
        
        # Oblicz zmiany cen
        delta = prices.diff()
        
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
        
        # Obliczanie EMA
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        
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
        
        # Oblicz średnie kroczące
        ma_short = prices.rolling(window=short_period).mean().iloc[-1]
        ma_long = prices.rolling(window=long_period).mean().iloc[-1]
        
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

def get_usd_balance():
    """Pobierz saldo USD."""
    try:
        accounts = client.get_accounts().accounts
        for account in accounts:
            if account.currency == 'USD':
                balance = account.available_balance
                if balance:
                    if isinstance(balance, dict):
                        return float(balance.get('value', 0))
                    else:
                        return float(balance.value)
        return 0
    except Exception as e:
        logging.error(f"Błąd podczas pobierania salda USD: {e}")
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
    if historical_data is None or historical_data.empty:
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
        
        # Pobierz ostatnie wartości
        current_price = historical_data['price'].iloc[-1]
        ma20 = historical_data['MA20'].iloc[-1]
        ma50 = historical_data['MA50'].iloc[-1]
        rsi = historical_data['RSI'].iloc[-1]
        
        # Sprawdź czy wartości nie są NaN
        if pd.isna(ma20) or pd.isna(ma50) or pd.isna(rsi):
            return "neutral"
        
        # Analiza trendu na podstawie wielu wskaźników
        if current_price > ma20 and ma20 > ma50 and rsi > 50:
            return "bullish"
        elif current_price < ma20 and ma20 < ma50 and rsi < 50:
            return "bearish"
        else:
            return "neutral"
    except Exception as e:
        logger.error(f"Błąd podczas analizy trendu: {e}")
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

def get_product_ticker(product_id):
    try:
        product = client.get_product(product_id)
        if product and hasattr(product, 'price'):
            return float(product.price)
        return None
    except Exception as e:
        logger.error(f"Błąd podczas pobierania aktualnej ceny dla {product_id}: {e}")
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
                        
                    current_price = get_product_ticker(pair)
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
        # Resetuj historię transakcji jeśli to nowy dzień
        if market_data[product_id]['trade_history']:
            last_trade_date = market_data[product_id]['trade_history'][-1]['timestamp'].date()
            if last_trade_date < today:
                market_data[product_id]['trade_history'] = []
                logging.info(f"Reset historii transakcji dla {product_id} - nowy dzień")
                save_trade_history()  # Zapisz zmiany
        
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
        save_trade_history()  # Zapisz zmiany po każdej transakcji
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
        save_trade_history()  # Zapisz zmiany po każdej transakcji
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
                # Sprawdź czy waluta jest obsługiwana
                if currency not in ['USDC', 'USD', 'BTC', 'ETH']:
                    continue
                    
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

                balances[currency] = {
                    'available': available,
                    'total': total,
                    'hold': hold,
                    'currency': currency,
                    'type': getattr(account, 'type', 'unknown'),
                    'active': getattr(account, 'active', True)
                }
        return balances
    except Exception as e:
        logger.error(f"Błąd podczas pobierania sald: {e}")
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
                            current_price = get_product_ticker(pair)
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
        # Wczytaj klucze API z pliku
        with open('cdp_api_key.json', 'r') as f:
            api_keys = json.load(f)
            api_key = api_keys['api_key_id']
            api_secret = api_keys['api_key_secret']
        
        # Utwórz klienta z kluczami API
        client = RESTClient(api_key=api_key, api_secret=api_secret)
        return client
    except Exception as e:
        logger.error(f"Błąd podczas tworzenia klienta: {str(e)}")
        return None

def save_trade_history():
    """Zapisuje historię transakcji do pliku."""
    try:
        with open('trade_history.json', 'w') as f:
            json.dump(market_data, f, default=str)
        logging.info("Zapisano historię transakcji")
    except Exception as e:
        logging.error(f"Błąd podczas zapisywania historii transakcji: {e}")

def load_trade_history():
    """Wczytuje historię transakcji z pliku."""
    try:
        if os.path.exists('trade_history.json'):
            with open('trade_history.json', 'r') as f:
                data = json.load(f)
                for pair in TRADING_PAIRS:
                    if pair in data:
                        market_data[pair]['trade_history'] = data[pair]['trade_history']
                        # Konwertuj stringi timestampów z powrotem na obiekty datetime
                        for trade in market_data[pair]['trade_history']:
                            trade['timestamp'] = datetime.fromisoformat(trade['timestamp'])
            logging.info("Wczytano historię transakcji")
    except Exception as e:
        logging.error(f"Błąd podczas wczytywania historii transakcji: {e}")

def get_product_candles(product_id, start, end, granularity):
    """Pobiera świeczki dla danej pary handlowej."""
    try:
        # Konwertuj daty na timestamp w sekundach
        start_timestamp = int(start.timestamp())
        end_timestamp = int(end.timestamp())
        
        # Użyj prawidłowej metody API
        response = client.get_candles(
            product_id=product_id,
            start=start_timestamp,
            end=end_timestamp,
            granularity=granularity
        )
        
        if not response or not hasattr(response, 'candles'):
            logger.warning(f"Brak danych świeczek dla {product_id}")
            return None
            
        return response.candles
        
    except Exception as e:
        logger.error(f"Błąd podczas pobierania świeczek dla {product_id}: {e}")
        return None

def calculate_market_sentiment(product_id):
    """Analizuje sentyment rynku na podstawie różnych wskaźników."""
    try:
        historical_data = get_historical_data(product_id)
        if historical_data is None or len(historical_data) < 50:
            return 0

        # Oblicz wskaźniki
        rsi = calculate_rsi(historical_data['price'])
        macd, signal, hist = calculate_macd(historical_data['price'])
        ma20, ma50 = calculate_moving_averages(historical_data['price'])
        volatility = calculate_volatility(historical_data)
        
        # Oblicz głębokość rynku
        market_depth = calculate_market_depth()
        if market_depth:
            depth_ratio = market_depth['ratio']
        else:
            depth_ratio = 1.0

        # Oblicz sentyment (wartość od -1 do 1)
        sentiment = 0
        
        # RSI contribution
        if rsi < 30:
            sentiment += 0.2
        elif rsi > 70:
            sentiment -= 0.2
            
        # MACD contribution
        if macd > signal:
            sentiment += 0.2
        else:
            sentiment -= 0.2
            
        # Moving Averages contribution
        if ma20 > ma50:
            sentiment += 0.2
        else:
            sentiment -= 0.2
            
        # Volatility contribution
        if volatility < 1.0:  # Niska zmienność
            sentiment += 0.1
        elif volatility > 3.0:  # Wysoka zmienność
            sentiment -= 0.1
            
        # Market depth contribution
        if depth_ratio > 1.2:  # Więcej kupujących
            sentiment += 0.2
        elif depth_ratio < 0.8:  # Więcej sprzedających
            sentiment -= 0.2
            
        return sentiment
        
    except Exception as e:
        logger.error(f"Błąd podczas obliczania sentymentu rynku: {e}")
        return 0

def calculate_position_size(product_id, current_price, sentiment):
    """Oblicza optymalny rozmiar pozycji na podstawie sentymentu i zarządzania ryzykiem."""
    try:
        # Pobierz dostępny kapitał dla pary
        available_capital = get_available_capital_for_pair(product_id)
        
        # Pobierz historię transakcji
        trade_history = market_data[product_id]['trade_history']
        
        # Oblicz dzienny P&L
        daily_pnl = 0
        today = datetime.utcnow().date()
        for trade in trade_history:
            trade_date = datetime.strptime(trade['timestamp'], '%Y-%m-%d %H:%M:%S.%f').date()
            if trade_date == today:
                if trade['type'] == 'sell':
                    daily_pnl += (trade['price'] - market_data[product_id]['last_buy_price']) * trade['amount']
                else:
                    daily_pnl -= trade['price'] * trade['amount']
        
        # Dostosuj rozmiar pozycji na podstawie sentymentu
        base_size = TRADE_VALUE_USDC
        
        # Zmniejsz rozmiar jeśli sentyment jest negatywny
        if sentiment < -0.5:
            base_size *= 0.5
        elif sentiment < 0:
            base_size *= 0.75
            
        # Zmniejsz rozmiar jeśli dzienny P&L jest ujemny
        if daily_pnl < 0:
            base_size *= 0.5
            
        # Sprawdź maksymalny rozmiar pozycji
        max_size = min(available_capital, MAX_POSITION_SIZE)
        
        # Zwróć mniejszą z wartości
        return min(base_size, max_size)
        
    except Exception as e:
        logger.error(f"Błąd podczas obliczania rozmiaru pozycji: {e}")
        return MIN_TRADE_SIZE_USDC

def implement_trailing_stop(product_id, current_price):
    """Implementuje trailing stop dla otwartych pozycji."""
    try:
        trade_history = market_data[product_id]['trade_history']
        if not trade_history:
            return False
            
        # Znajdź ostatnią transakcję kupna
        last_buy = None
        for trade in reversed(trade_history):
            if trade['type'] == 'buy':
                last_buy = trade
                break
                
        if not last_buy:
            return False
            
        # Oblicz zysk
        profit_percent = (current_price - last_buy['price']) / last_buy['price'] * 100
        
        # Jeśli zysk przekracza trailing stop, zaktualizuj stop loss
        if profit_percent > TRAILING_STOP_PERCENT:
            new_stop_price = current_price * (1 - TRAILING_STOP_PERCENT/100)
            
            # Jeśli cena spadła poniżej nowego stop loss, sprzedaj
            if current_price < new_stop_price:
                amount = last_buy['amount']
                place_sell_order(product_id, amount, current_price)
                logger.info(f"Wykonano trailing stop dla {product_id} przy cenie {current_price}")
                return True
                
        return False
        
    except Exception as e:
        logger.error(f"Błąd podczas implementacji trailing stop: {e}")
        return False

def analyze_correlation():
    """Analizuje korelację między kryptowalutami."""
    try:
        correlations = {}
        for pair in TRADING_PAIRS:
            historical_data = get_historical_data(pair)
            if historical_data is not None:
                correlations[pair] = historical_data['price']
                
        # Oblicz korelację między parami
        correlation_matrix = pd.DataFrame(correlations).corr()
        
        # Znajdź pary z wysoką korelacją
        high_correlation_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if abs(correlation_matrix.iloc[i,j]) > 0.7:  # Korelacja powyżej 0.7
                    high_correlation_pairs.append({
                        'pair1': correlation_matrix.columns[i],
                        'pair2': correlation_matrix.columns[j],
                        'correlation': correlation_matrix.iloc[i,j]
                    })
                    
        return high_correlation_pairs
        
    except Exception as e:
        logger.error(f"Błąd podczas analizy korelacji: {e}")
        return []

def analyze_market_conditions():
    """Analizuje ogólne warunki rynkowe."""
    try:
        market_conditions = {
            'volatility': {},
            'trend': {},
            'sentiment': {},
            'correlation': [],
            'risk_level': 'NORMAL'
        }
        
        # Analizuj każdą parę
        for pair in TRADING_PAIRS:
            historical_data = get_historical_data(pair)
            if historical_data is not None and not historical_data.empty:
                # Oblicz zmienność
                volatility = calculate_volatility(historical_data)
                market_conditions['volatility'][pair] = volatility
                
                # Analizuj trend
                trend = analyze_market_trend(historical_data)
                market_conditions['trend'][pair] = trend
                
                # Oblicz sentyment
                sentiment = calculate_market_sentiment(pair)
                market_conditions['sentiment'][pair] = sentiment
                
        # Analizuj korelacje
        market_conditions['correlation'] = analyze_correlation()
        
        # Określ poziom ryzyka
        high_volatility_count = sum(1 for v in market_conditions['volatility'].values() if v > 3.0)
        negative_sentiment_count = sum(1 for s in market_conditions['sentiment'].values() if s < -0.5)
        
        if high_volatility_count >= 2 or negative_sentiment_count >= 2:
            market_conditions['risk_level'] = 'HIGH'
        elif high_volatility_count >= 1 or negative_sentiment_count >= 1:
            market_conditions['risk_level'] = 'MEDIUM'
            
        return market_conditions
        
    except Exception as e:
        logger.error(f"Błąd podczas analizy warunków rynkowych: {e}")
        return None

def clear_trade_history():
    """Czyści historię transakcji dla wszystkich par."""
    try:
        for pair in TRADING_PAIRS:
            if pair in market_data:
                market_data[pair]['trade_history'] = []
                market_data[pair]['last_buy_price'] = None
                market_data[pair]['last_sell_price'] = None
                market_data[pair]['highest_price'] = None
        save_trade_history()
        logger.info("Wyczyszczono historię transakcji")
    except Exception as e:
        logger.error(f"Błąd podczas czyszczenia historii transakcji: {e}")

def implement_risk_management():
    """Implementuje zarządzanie ryzykiem dla całego portfela."""
    try:
        # Pobierz wszystkie salda
        balances = get_all_balances()
        
        # Oblicz całkowitą wartość portfela
        total_value = 0
        for currency, data in balances.items():
            if currency in ['USDC', 'USD']:
                total_value += float(data['available'])
            elif currency in ['BTC', 'ETH']:
                # Pobierz aktualną cenę
                price = get_product_ticker(f"{currency}-USDC")
                if price:
                    total_value += float(data['available']) * float(price)
                    
        # Sprawdź dzienny P&L
        daily_pnl = 0
        today = datetime.utcnow().date()
        
        # Najpierw sprawdź i zresetuj historię transakcji dla wszystkich par
        for pair in TRADING_PAIRS:
            if pair not in market_data:
                continue
                
            trade_history = market_data[pair]['trade_history']
            if trade_history:
                # Znajdź najnowszą transakcję
                last_trade = max(trade_history, key=lambda x: x['timestamp'])
                last_trade_date = last_trade['timestamp'].date()
                
                # Jeśli ostatnia transakcja była wcześniej niż dzisiaj, zresetuj historię
                if last_trade_date < today:
                    market_data[pair]['trade_history'] = []
                    market_data[pair]['last_buy_price'] = None
                    market_data[pair]['last_sell_price'] = None
                    market_data[pair]['highest_price'] = None
                    logger.info(f"Reset historii transakcji dla {pair} - nowy dzień")
                    save_trade_history()  # Zapisz zmiany
        
        # Teraz oblicz P&L tylko dla dzisiejszych transakcji
        for pair in TRADING_PAIRS:
            if pair not in market_data:
                continue
                
            trade_history = market_data[pair]['trade_history']
            
            for trade in trade_history:
                if not isinstance(trade['timestamp'], datetime):
                    continue
                    
                trade_date = trade['timestamp'].date()
                if trade_date == today:
                    if trade['type'] == 'sell' and 'last_buy_price' in market_data[pair]:
                        profit = (float(trade['price']) - float(market_data[pair]['last_buy_price'])) * float(trade['amount'])
                        daily_pnl += profit
                        logger.info(f"Transakcja sprzedaży dla {pair}: {profit:.2f} USDC")
                    elif trade['type'] == 'buy':
                        cost = float(trade['price']) * float(trade['amount'])
                        daily_pnl -= cost
                        logger.info(f"Transakcja kupna dla {pair}: -{cost:.2f} USDC")
        
        # Loguj aktualny P&L
        logger.info(f"Aktualny dzienny P&L: {daily_pnl:.2f} USDC")
        logger.info(f"Limit P&L: {total_value * 0.05:.2f} USDC")
        
        # Jeśli dzienny P&L jest ujemny i przekracza 5% wartości portfela, zatrzymaj handel
        if daily_pnl < 0 and abs(daily_pnl) > total_value * 0.05:
            logger.warning(f"Dzienny P&L przekroczył limit: {daily_pnl:.2f} USDC (limit: {total_value * 0.05:.2f} USDC)")
            # Wyczyść historię transakcji aby zacząć od nowa
            clear_trade_history()
            return False
            
        # Sprawdź maksymalną ekspozycję
        total_exposure = check_total_exposure()
        if total_exposure > MAX_TOTAL_EXPOSURE:
            logger.warning(f"Maksymalna ekspozycja przekroczona: {total_exposure:.2f} USDC (limit: {MAX_TOTAL_EXPOSURE:.2f} USDC)")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Błąd podczas zarządzania ryzykiem: {e}")
        return False

def main():
    """Główna funkcja bota"""
    try:
        logger.info("=== ROZPOCZYNAM DZIAŁANIE BOTA ===")
        
        # Sprawdź dostępność par handlowych
        logger.info("Sprawdzam dostępność par handlowych...")
        for pair in TRADING_PAIRS:
            try:
                client.get_product(pair)
                logger.info(f"Para {pair} jest dostępna")
            except Exception as e:
                logger.error(f"Para {pair} jest niedostępna: {e}")
                return
                
        # Załaduj historię transakcji
        load_trade_history()
        
        # Główna pętla handlowa
        while True:
            try:
                logger.info("\n=== NOWA ITERACJA PĘTLI HANDLOWEJ ===")
                
                # Sprawdź salda
                logger.info("\n=== AKTUALNE SALDA ===")
                usdc_balance = get_usdc_balance()
                usd_balance = get_usd_balance()
                logger.info(f"USDC: {usdc_balance}")
                logger.info(f"USD: {usd_balance}")
                
                # Analizuj warunki rynkowe
                market_conditions = analyze_market_conditions()
                if market_conditions:
                    logger.info("\n=== WARUNKI RYNKOWE ===")
                    logger.info(f"Poziom ryzyka: {market_conditions['risk_level']}")
                    
                    # Sprawdź zarządzanie ryzykiem
                    if not implement_risk_management():
                        logger.warning("Zatrzymuję handel ze względu na zarządzanie ryzykiem")
                        time.sleep(300)  # Czekaj 5 minut przed następną iteracją
                        continue
                        
                    # Analizuj każdą parę
                    for pair in TRADING_PAIRS:
                        logger.info(f"\n=== ANALIZA PARY {pair} ===")
                        
                        # Pobierz dane historyczne
                        historical_data = get_historical_data(pair)
                        if historical_data is None:
                            continue
                            
                        # Pobierz aktualną cenę
                        current_price = get_product_ticker(pair)
                        if not current_price:
                            continue
                            
                        logger.info(f"Aktualna cena {pair}: {current_price}")
                        
                        # Oblicz wskaźniki techniczne
                        logger.info(f"\n=== WSKAŹNIKI TECHNICZNE DLA {pair} ===")
                        rsi = calculate_rsi(historical_data['price'])
                        macd, signal, hist = calculate_macd(historical_data['price'])
                        ma20, ma50 = calculate_moving_averages(historical_data['price'])
                        volatility = calculate_volatility(historical_data)
                        
                        logger.info(f"RSI: {rsi:.2f}")
                        logger.info(f"MACD: {macd:.2f}")
                        logger.info(f"Signal: {signal:.2f}")
                        logger.info(f"Histogram: {hist:.2f}")
                        logger.info(f"MA20: {ma20:.2f}")
                        logger.info(f"MA50: {ma50:.2f}")
                        logger.info(f"Zmienność ceny: {volatility:.2f}%")
                        
                        # Oblicz sentyment rynku
                        sentiment = calculate_market_sentiment(pair)
                        
                        # Sprawdź trailing stop
                        if implement_trailing_stop(pair, current_price):
                            continue
                            
                        # Sprawdź warunki kupna
                        if pair.endswith('USDC'):
                            logger.info(f"\n=== SPRAWDZANIE WARUNKÓW KUPNA {pair.split('-')[0]} ZA USDC ===")
                            
                            # Sprawdź podstawowe warunki
                            rsi_condition = rsi < 50
                            macd_condition = macd > signal
                            hist_condition = hist > 0
                            ma_condition = current_price > ma20
                            
                            logger.info(f"RSI < 50: {rsi_condition}")
                            logger.info(f"MACD > Signal: {macd_condition}")
                            logger.info(f"Histogram > 0: {hist_condition}")
                            logger.info(f"Cena > MA20: {ma_condition}")
                            
                            # Sprawdź dodatkowe warunki
                            sentiment_condition = sentiment > 0
                            volatility_condition = volatility < 3.0
                            risk_condition = market_conditions['risk_level'] != 'HIGH'
                            
                            # Jeśli wszystkie warunki są spełnione
                            if (rsi_condition and macd_condition and hist_condition and ma_condition and 
                                sentiment_condition and volatility_condition and risk_condition):
                                
                                logger.info("!!! SYGNAŁ KUPNA - PRÓBA WEJŚCIA W POZYCJĘ !!!")
                                
                                # Oblicz rozmiar pozycji
                                position_size = calculate_position_size(pair, current_price, sentiment)
                                
                                # Złóż zlecenie kupna
                                order = place_buy_order(pair, position_size, current_price)
                                if order and order.get('success'):
                                    logger.info(f"Złożono zlecenie kupna: {order}")
                                    
                        # Sprawdź warunki sprzedaży
                        trade_history = market_data[pair]['trade_history']
                        if trade_history:
                            last_buy = None
                            for trade in reversed(trade_history):
                                if trade['type'] == 'buy':
                                    last_buy = trade
                                    break
                                    
                            if last_buy:
                                # Oblicz zysk
                                profit_percent = (current_price - last_buy['price']) / last_buy['price'] * 100
                                
                                # Sprawdź warunki sprzedaży
                                if (profit_percent >= TAKE_PROFIT_PERCENT or 
                                    profit_percent <= -STOP_LOSS_PERCENT or
                                    sentiment < -0.5):
                                    
                                    logger.info(f"!!! SYGNAŁ SPRZEDAŻY - PRÓBA WYJŚCIA Z POZYCJI !!!")
                                    logger.info(f"Zysk: {profit_percent:.2f}%")
                                    
                                    # Sprzedaj całą pozycję
                                    amount = last_buy['amount']
                                    order = place_sell_order(pair, amount, current_price)
                                    if order and order.get('success'):
                                        logger.info(f"Złożono zlecenie sprzedaży: {order}")
                                        
                # Zapisz historię transakcji
                save_trade_history()
                
                # Czekaj przed następną iteracją
                logger.info("\n=== CZEKAM 60 SEKUND PRZED NASTĘPNĄ ITERACJĄ ===")
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"Błąd w głównej pętli: {e}")
                time.sleep(60)
                
    except Exception as e:
        logger.error(f"Krytyczny błąd: {e}")
        
if __name__ == "__main__":
    main()




