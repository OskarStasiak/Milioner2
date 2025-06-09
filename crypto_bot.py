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

# Ładowanie zmiennych środowiskowych
load_dotenv('production.env')

# Konfiguracja loggera
logging.basicConfig(
    filename='crypto_bot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Zmienne globalne
TRADING_PAIRS = ['ETH-USDC', 'BTC-USDC', 'SOL-USDC', 'DOGE-USDC', 'XRP-USDC', 'MATIC-USDC', 'LINK-USDC']
market_data = {}
ws = None

# Pobierz zmienne z pliku .env
API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')
MAX_TOTAL_EXPOSURE = float(os.getenv('MAX_TOTAL_EXPOSURE', 1000))
MAX_LOSS_PERCENT = float(os.getenv('MAX_LOSS_PERCENT', 2))
STOP_LOSS_PERCENT = float(os.getenv('STOP_LOSS_PERCENT', 1))
TAKE_PROFIT_PERCENT = float(os.getenv('TAKE_PROFIT_PERCENT', 2))
TRAILING_STOP_PERCENT = float(os.getenv('TRAILING_STOP_PERCENT', 0.5))
MIN_TIME_BETWEEN_TRADES = int(os.getenv('MIN_TIME_BETWEEN_TRADES', 300))
MIN_HOLDING_TIME = int(os.getenv('MIN_HOLDING_TIME', 600))
MAX_VOLATILITY = float(os.getenv('MAX_VOLATILITY', 5.0))
TRADE_INTERVAL = int(os.getenv('TRADE_INTERVAL', 30))

# Konfiguracja API
with open('cdp_api_key.json', 'r') as f:
    api_keys = json.load(f)
    API_KEY = api_keys['api_key_id']
    API_SECRET = api_keys['api_key_secret']

# Parametry handlowe
TRADE_VALUE_USDC = float(os.getenv('TRADE_VALUE_USDC', 50))  # Zmniejszona wartość pojedynczej transakcji
MAX_POSITION_SIZE = float(os.getenv('MAX_POSITION_SIZE', 0.05))  # Zwiększony maksymalny rozmiar pozycji
PRICE_THRESHOLD_BUY = float(os.getenv('PRICE_THRESHOLD_BUY', 2500))  # Obniżony próg kupna
PRICE_THRESHOLD_SELL = float(os.getenv('PRICE_THRESHOLD_SELL', 2800))  # Podwyższony próg sprzedaży
MIN_PROFIT_PERCENT = float(os.getenv('MIN_PROFIT_PERCENT', 0.3))  # Zmniejszony minimalny zysk

# Parametry podwajania zysków
PROFIT_DOUBLING_DAYS = 2  # Co 2 dni podwajamy zyski
INITIAL_PROFIT_TARGET = 1.0  # Początkowy cel zysku w procentach
MAX_PROFIT_TARGET = 15.0  # Maksymalny cel zysku w procentach
PROFIT_MULTIPLIER = 2.5  # Mnożnik do podwajania zysków

# Parametry zarządzania kapitałem
CAPITAL_ALLOCATION = {
    'ETH-USDC': 0.20,  # Zwiększony udział ETH
    'BTC-USDC': 0.15,
    'SOL-USDC': 0.15,
    'DOGE-USDC': 0.10,
    'XRP-USDC': 0.15,
    'MATIC-USDC': 0.15,
    'LINK-USDC': 0.10,
    'RESERVE': 0.0
}

MIN_TRADE_SIZE_USDC = 5  # Zmniejszona minimalna wielkość zlecenia
MAX_TRADES_PER_DAY = 20   # Zwiększona maksymalna liczba transakcji dziennie

# Parametry zarządzania zyskami
MIN_PROFIT_TARGET = 0.5  # Minimalny cel zysku w procentach
MAX_PROFIT_TARGET = 10.0  # Maksymalny cel zysku w procentach
VOLATILITY_MULTIPLIER = 2.5  # Mnożnik zmienności do obliczania celu zysku
TREND_STRENGTH_MULTIPLIER = 2.0  # Mnożnik siły trendu do obliczania celu zysku

# Minimalny rozmiar zlecenia dla każdej pary
MIN_ORDER_SIZE_USDC = {
    'ETH-USDC': 10.0,
    'BTC-USDC': 10.0,
    'SOL-USDC': 10.0,
    'DOGE-USDC': 10.0,
    'XRP-USDC': 10.0,
    'MATIC-USDC': 10.0,
    'LINK-USDC': 10.0
}

# Parametry handlowe dla różnych par
TRADING_PARAMS = {
    'ETH-USDC': {
        'rsi_threshold': 45,  # Zmniejszony próg RSI
        'macd_threshold': -0.1,  # Zmniejszony próg MACD
        'ma_threshold': -0.1,
        'volatility_threshold': 0.8,
        'profit_target': 0.8,
        'stop_loss': 0.4,
        'position_size': 0.20
    },
    'BTC-USDC': {
        'rsi_threshold': 45,
        'macd_threshold': -0.1,
        'ma_threshold': -0.1,
        'volatility_threshold': 0.8,
        'profit_target': 0.8,
        'stop_loss': 0.4,
        'position_size': 0.15
    },
    'SOL-USDC': {
        'rsi_threshold': 45,
        'macd_threshold': -0.1,
        'ma_threshold': -0.1,
        'volatility_threshold': 1.0,
        'profit_target': 1.0,
        'stop_loss': 0.5,
        'position_size': 0.15
    },
    'DOGE-USDC': {
        'rsi_threshold': 50,
        'macd_threshold': -0.1,
        'ma_threshold': -0.1,
        'volatility_threshold': 1.5,
        'profit_target': 1.5,
        'stop_loss': 0.8,
        'position_size': 0.10
    },
    'XRP-USDC': {
        'rsi_threshold': 45,
        'macd_threshold': -0.1,
        'ma_threshold': -0.1,
        'volatility_threshold': 1.0,
        'profit_target': 1.0,
        'stop_loss': 0.5,
        'position_size': 0.15
    },
    'MATIC-USDC': {
        'rsi_threshold': 45,
        'macd_threshold': -0.1,
        'ma_threshold': -0.1,
        'volatility_threshold': 1.0,
        'profit_target': 1.0,
        'stop_loss': 0.5,
        'position_size': 0.15
    },
    'LINK-USDC': {
        'rsi_threshold': 45,
        'macd_threshold': -0.1,
        'ma_threshold': -0.1,
        'volatility_threshold': 1.0,
        'profit_target': 1.0,
        'stop_loss': 0.5,
        'position_size': 0.10
    }
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
        'highest_price': None,
        'historical_data': None  # Dodajemy pole historical_data
    }
    logging.info(f"Inicjalizacja danych dla {pair} - historia transakcji wyczyszczona")

def on_ws_message(data):
    """Obsługa wiadomości z WebSocket."""
    try:
        if data['type'] == 'ticker':
            product_id = data.get('product_id')
            if product_id not in TRADING_PAIRS:
                return
                
            price = float(data.get('price', 0))
            if price > 0:
                market_data[product_id]['current_price'] = price
                market_data[product_id]['price_history'].append({
                    'timestamp': datetime.utcnow(),
                    'price': price
                })
                # Zachowaj tylko ostatnie 1000 cen
                if len(market_data[product_id]['price_history']) > 1000:
                    market_data[product_id]['price_history'].pop(0)
                logging.info(f"Aktualna cena {product_id}: {price}")
            
        elif data['type'] == 'snapshot':
            product_id = data.get('product_id')
            if product_id not in TRADING_PAIRS:
                return
                
            market_data[product_id]['order_book']['bids'] = data.get('bids', [])
            market_data[product_id]['order_book']['asks'] = data.get('asks', [])
            logging.info(f"Zaktualizowano książkę zleceń dla {product_id}: {len(market_data[product_id]['order_book']['bids'])} bidów, {len(market_data[product_id]['order_book']['asks'])} asków")
            
        elif data['type'] == 'l2update':
            product_id = data.get('product_id')
            if product_id not in TRADING_PAIRS:
                return
                
            changes = data.get('changes', [])
            for change in changes:
                side, price, size = change
                if side == 'buy':
                    market_data[product_id]['order_book']['bids'] = [b for b in market_data[product_id]['order_book']['bids'] if b[0] != price]
                    if float(size) > 0:
                        market_data[product_id]['order_book']['bids'].append([price, size])
                else:
                    market_data[product_id]['order_book']['asks'] = [a for a in market_data[product_id]['order_book']['asks'] if a[0] != price]
                    if float(size) > 0:
                        market_data[product_id]['order_book']['asks'].append([price, size])
            
            # Sortowanie książki zleceń
            market_data[product_id]['order_book']['bids'].sort(key=lambda x: float(x[0]), reverse=True)
            market_data[product_id]['order_book']['asks'].sort(key=lambda x: float(x[0]))
            
    except Exception as e:
        logging.error(f"Błąd podczas przetwarzania wiadomości WebSocket: {e}")

def calculate_market_depth(product_id):
    """Oblicza głębokość rynku na podstawie książki zleceń."""
    try:
        if product_id not in market_data:
            return None
            
        total_bids = sum(float(bid[1]) for bid in market_data[product_id]['order_book']['bids'][:10])  # Top 10 bidów
        total_asks = sum(float(ask[1]) for ask in market_data[product_id]['order_book']['asks'][:10])  # Top 10 asków
        
        return {
            'bids_volume': total_bids,
            'asks_volume': total_asks,
            'ratio': total_bids / total_asks if total_asks > 0 else 0
        }
    except Exception as e:
        logging.error(f"Błąd podczas obliczania głębokości rynku dla {product_id}: {e}")
        return None

def calculate_volatility(prices):
    """Oblicza zmienność ceny."""
    try:
        if len(prices) < 2:
            return 0.0
            
        # Oblicz procentowe zmiany
        returns = np.diff(prices) / prices[:-1]
        
        # Oblicz odchylenie standardowe
        volatility = np.std(returns) * 100  # Konwersja na procenty
        
        return float(volatility)
        
    except Exception as e:
        logging.error(f"Błąd podczas obliczania zmienności: {e}")
        return 0.0

def calculate_rsi(prices):
    """Oblicza wskaźnik RSI."""
    try:
        if len(prices) < 14:
            return None
        
        # Oblicz zmiany cen
        deltas = np.diff(prices)
        seed = deltas[:14]
        up = seed[seed >= 0].sum()/14
        down = -seed[seed < 0].sum()/14
        rs = up/down if down != 0 else 0
        rsi = np.zeros_like(prices)
        rsi[:14] = 100. - 100./(1. + rs)
        
        # Oblicz pozostałe wartości RSI
        for i in range(14, len(prices)):
            delta = deltas[i - 1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
                
            up = (up*13 + upval)/14
            down = (down*13 + downval)/14
            rs = up/down if down != 0 else 0
            rsi[i] = 100. - 100./(1. + rs)
            
        return float(rsi[-1])  # Zwróć ostatnią wartość RSI jako float
        
    except Exception as e:
        logging.error(f"Błąd podczas obliczania RSI: {e}")
        return None

def calculate_macd(prices):
    """Oblicza wskaźnik MACD."""
    try:
        if len(prices) < 26:
            return None, None, None
        
        # Oblicz EMA
        ema12 = pd.Series(prices).ewm(span=12, adjust=False).mean()
        ema26 = pd.Series(prices).ewm(span=26, adjust=False).mean()
        
        # Oblicz MACD i linię sygnału
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal
        
        # Pobierz ostatnie wartości i konwertuj na float
        macd_last = float(macd.iloc[-1])
        signal_last = float(signal.iloc[-1])
        hist_last = float(hist.iloc[-1])
        
        # Sprawdź czy wartości nie są NaN
        if pd.isna(macd_last) or pd.isna(signal_last) or pd.isna(hist_last):
            return None, None, None
            
        return macd_last, signal_last, hist_last
        
    except Exception as e:
        logging.error(f"Błąd podczas obliczania MACD: {e}")
        return None, None, None

def calculate_moving_averages(prices):
    """Oblicza średnie kroczące."""
    try:
        if len(prices) < 50:
            return None, None
        
        # Konwertuj na Series jeśli to nie jest już Series
        if not isinstance(prices, pd.Series):
            prices = pd.Series(prices)
        
        # Oblicz średnie kroczące
        ma20 = prices.rolling(window=20).mean()
        ma50 = prices.rolling(window=50).mean()
        
        # Sprawdź czy mamy wystarczająco danych
        if ma20.isna().all() or ma50.isna().all():
            return None, None
            
        # Pobierz ostatnie wartości i konwertuj na float
        ma20_last = float(ma20.iloc[-1])
        ma50_last = float(ma50.iloc[-1])
        
        # Sprawdź czy wartości nie są NaN
        if pd.isna(ma20_last) or pd.isna(ma50_last):
            return None, None
            
        return ma20_last, ma50_last
        
    except Exception as e:
        logging.error(f"Błąd podczas obliczania średnich kroczących: {e}")
        return None, None

def get_available_balance(currency):
    """Pobiera dostępne saldo dla danej waluty, ignorując środki w stakingu."""
    try:
        accounts = client.get_accounts().accounts
        for account in accounts:
            if account.currency == currency:
                if hasattr(account, 'available_balance'):
                    if isinstance(account.available_balance, dict):
                        return float(account.available_balance.get('value', 0))
                    else:
                        return float(account.available_balance.value)
        return 0
    except Exception as e:
        logger.error(f"Błąd podczas pobierania salda {currency}: {e}")
        return 0

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
    """Sprawdza czy należy wykonać transakcję na podstawie wskaźników technicznych."""
    try:
        # Konwertuj dane na DataFrame
        df = pd.DataFrame(historical_data)
            
        # Oblicz wskaźniki techniczne
        rsi = calculate_rsi(df['price'].values)
        macd, macd_signal, macd_hist = calculate_macd(df['price'].values)
        ma_short, ma_long = calculate_moving_averages(df['price'].values)
        
        # Sprawdź czy wszystkie wskaźniki są dostępne
        if rsi is None or macd is None or macd_signal is None or ma_short is None or ma_long is None:
            return False
            
        # Pobierz ostatnie wartości cen
        last_price = float(df['price'].iloc[-1])
        prev_price = float(df['price'].iloc[-2])
        
        # Sprawdź czy cena rośnie
        if last_price > prev_price:
            # Sprawdź czy RSI jest w odpowiednim zakresie
            if 30 <= rsi <= 70:
                # Sprawdź czy MACD jest dodatni
                if macd > 0:
                    # Sprawdź czy MACD rośnie
                    if macd > macd_signal:
                        # Sprawdź czy MACD rośnie szybciej niż sygnał
                        if (macd - macd_signal) > 0:
            return True
        return False
        
    except Exception as e:
        logger.error(f"Błąd podczas analizy warunków handlowych: {e}")
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
        
        # Użyj prawidłowej metody API
        response = client.get_candles(
            product_id=product_id,
            start=start_timestamp,
            end=end_timestamp,
            granularity="ONE_HOUR"
        )
        
        if not response or not hasattr(response, 'candles'):
            logger.warning(f"Brak danych historycznych dla {product_id}")
            return None
            
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
                logger.warning(f"Pominięto nieprawidłową świeczkę dla {product_id}: {e}")
                continue
        
        if not data:
            logger.warning(f"Brak prawidłowych danych historycznych dla {product_id}")
            return None
            
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        logger.info(f"Pobrano {len(df)} świeczek dla {product_id}")
        return df
        
    except Exception as e:
        logger.error(f"Błąd podczas pobierania historycznych danych dla {product_id}: {e}")
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
    try:
        if historical_data is None or (isinstance(historical_data, pd.DataFrame) and historical_data.empty) or len(historical_data) < 24:
        return "neutral"
    
        # Sprawdź czy historical_data jest DataFrame i konwertuj na numpy array
        if isinstance(historical_data, pd.DataFrame):
            prices = historical_data['price'].values
        else:
            prices = np.array(historical_data)
            
        if len(prices) == 0:
            return "neutral"
            
        # Upewnij się, że wszystkie wartości są float
        current_price = float(prices[-1])
        
        # Oblicz wskaźniki techniczne
        ma_short, ma_long = calculate_moving_averages(prices)
        rsi = calculate_rsi(prices)
        
        # Sprawdź czy wartości nie są None
        if ma_short is None or ma_long is None or rsi is None:
            return "neutral"
        
        # Konwertuj wartości na float
        ma_short = float(ma_short)
        ma_long = float(ma_long)
        rsi = float(rsi)
        
        # Analiza trendu na podstawie wielu wskaźników
        trend_bullish = current_price > ma_short and ma_short > ma_long and rsi > 50
        trend_bearish = current_price < ma_short and ma_short < ma_long and rsi < 50
        
        if trend_bullish:
            return "bullish"
        elif trend_bearish:
            return "bearish"
        else:
            return "neutral"
            
    except Exception as e:
        logger.error(f"Błąd podczas analizy trendu: {e}")
        return "neutral"

def suggest_thresholds(current_price, trend, predicted_price=None):
    """Sugeruj progi kupna/sprzedaży na podstawie analizy AI."""
    try:
        # Upewnij się, że trend jest stringiem
        if not isinstance(trend, str):
            trend = "neutral"
            
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
    """Pobiera aktualną cenę dla danej pary handlowej."""
    try:
        # Użyj prawidłowej metody API dla Coinbase Advanced
        product = client.get_product(product_id)
        if product and hasattr(product, 'price'):
            return float(product.price)
        elif product and hasattr(product, 'last_trade_price'):
            return float(product.last_trade_price)
        return None
    except Exception as e:
        logger.error(f"Błąd podczas pobierania aktualnej ceny dla {product_id}: {e}")
        return None

def check_total_exposure():
    """Sprawdza całkowitą ekspozycję portfela."""
    try:
        total_exposure = 0
        for pair in TRADING_PAIRS:
            if pair not in market_data:
                        continue
                        
            trade_history = market_data[pair]['trade_history']
            if not trade_history:
                        continue
                        
            # Znajdź ostatnią transakcję kupna
            last_buy = None
            for trade in reversed(trade_history):
                if trade['type'] == 'buy':
                    last_buy = trade
                    break
            
            if last_buy:
                # Oblicz wartość pozycji
                position_value = float(last_buy['amount']) * float(last_buy['price'])
                    total_exposure += position_value
                    
        return total_exposure
        
    except Exception as e:
        logger.error(f"Błąd podczas sprawdzania całkowitej ekspozycji: {e}")
        return 0

def can_open_new_position():
    """Sprawdza czy można otworzyć nową pozycję."""
    try:
        # Sprawdź całkowitą ekspozycję
        total_exposure = check_total_exposure()
        if total_exposure >= MAX_TOTAL_EXPOSURE:
            logger.warning(f"Przekroczono maksymalną ekspozycję: {total_exposure:.2f} USDC (limit: {MAX_TOTAL_EXPOSURE:.2f} USDC)")
            return False
            
        # Sprawdź dzienny P&L
        daily_pnl = check_daily_pnl()
        pnl_limit = calculate_pnl_limit()
        if daily_pnl < 0 and abs(daily_pnl) > pnl_limit:
            logger.warning(f"Przekroczono dzienny limit straty: {daily_pnl:.2f} USDC (limit: {pnl_limit:.2f} USDC)")
            return False
            
        # Sprawdź liczbę otwartych pozycji
        open_positions = 0
        for pair in TRADING_PAIRS:
            if pair in market_data and market_data[pair].get('in_position', False):
                open_positions += 1
                
        if open_positions >= MAX_OPEN_POSITIONS:
            logger.warning(f"Przekroczono maksymalną liczbę otwartych pozycji: {open_positions} (limit: {MAX_OPEN_POSITIONS})")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Błąd podczas sprawdzania możliwości otwarcia nowej pozycji: {e}")
        return False

def calculate_profit_target(product_id, historical_data):
    """Oblicza dynamiczny cel zysku na podstawie analizy rynku."""
    try:
        if historical_data is None or (isinstance(historical_data, pd.DataFrame) and historical_data.empty) or len(historical_data) < 24:
            return MIN_PROFIT_TARGET
            
        # Sprawdź czy historical_data jest DataFrame i konwertuj na numpy array
        if isinstance(historical_data, pd.DataFrame):
            prices = historical_data['price'].values
        else:
            prices = np.array(historical_data)
            
        if len(prices) == 0:
            return MIN_PROFIT_TARGET
            
        # Oblicz zmienność ceny
        volatility = calculate_volatility(prices)
        
        # Oblicz siłę trendu
        rsi = calculate_rsi(prices)
        macd, signal, hist = calculate_macd(prices)
        ma_short, ma_long = calculate_moving_averages(prices)
        
        if macd is None or ma_short is None or rsi is None:
            return MIN_PROFIT_TARGET
            
        # Określ siłę trendu
        trend_strength = 1.0
        current_price = float(prices[-1])
        ma_short = float(ma_short)
        ma_long = float(ma_long)
        
        if bool(current_price > ma_short and ma_short > ma_long):
            trend_strength = TREND_STRENGTH_MULTIPLIER
        elif bool(current_price < ma_short and ma_short < ma_long):
            trend_strength = 0.5
            
        # Oblicz cel zysku na podstawie zmienności i trendu
        profit_target = float(volatility) * VOLATILITY_MULTIPLIER * trend_strength
        
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
            
        # Konwertuj wartości na float
        current_price = float(current_price)
        last_buy_price = float(last_buy_price)
            
        # Oblicz procentową zmianę ceny
        price_change_percent = ((current_price - last_buy_price) / last_buy_price) * 100
        
        # Pobierz dynamiczny cel zysku
        profit_target = calculate_profit_target(product_id, historical_data)
        
        # Sprawdź czy osiągnięto cel zysku
        if price_change_percent >= profit_target:
            logging.info(f"Osiągnięto cel zysku {profit_target:.2f}% dla {product_id} (zmiana: {price_change_percent:.2f}%)")
            return True
            
        # Sprawdź czy trend się odwraca
        if historical_data is None or (isinstance(historical_data, pd.DataFrame) and historical_data.empty):
            return False
        
        if isinstance(historical_data, pd.DataFrame):
            prices = historical_data['price'].values
        else:
            prices = np.array(historical_data)
            
        if len(prices) == 0:
        return False
        
        rsi = calculate_rsi(prices)
        macd, signal, hist = calculate_macd(prices)

        if rsi is None or macd is None or hist is None:
            return False
            
        # Konwertuj wartości na float
        rsi = float(rsi)
        hist = float(hist)
        
        if bool(price_change_percent > 0 and rsi > 70 and hist < 0):
            logging.info(f"Trend się odwraca - realizacja zysku {price_change_percent:.2f}% dla {product_id}")
            return True
            
        return False
        
    except Exception as e:
        logging.error(f"Błąd podczas sprawdzania realizacji zysku: {e}")
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
                volatility = calculate_volatility(historical_data['price'].values)
                market_conditions['volatility'][pair] = float(volatility)
                
                # Analizuj trend
                trend = analyze_market_trend(historical_data)
                if isinstance(trend, str):  # Upewnij się, że trend jest stringiem
                    market_conditions['trend'][pair] = trend
                else:
                    market_conditions['trend'][pair] = "neutral"
                
                # Oblicz sentyment
                sentiment = calculate_market_sentiment(pair)
                market_conditions['sentiment'][pair] = float(sentiment)
                
        # Analizuj korelacje
        market_conditions['correlation'] = analyze_correlation()
        
        # Określ poziom ryzyka
        high_volatility_count = sum(1 for v in market_conditions['volatility'].values() if float(v) > 3.0)
        negative_sentiment_count = sum(1 for s in market_conditions['sentiment'].values() if float(s) < -0.5)
        
        if high_volatility_count >= 2 or negative_sentiment_count >= 2:
            market_conditions['risk_level'] = 'HIGH'
        elif high_volatility_count >= 1 or negative_sentiment_count >= 1:
            market_conditions['risk_level'] = 'MEDIUM'
            
        return market_conditions
        
    except Exception as e:
        logger.error(f"Błąd podczas analizy warunków rynkowych: {e}")
        return None

def check_buy_conditions(product_id):
    """Sprawdza warunki kupna dla danej pary."""
    try:
        if product_id not in market_data:
            return True
            
        # Sprawdź warunki handlowe
        if not check_trading_conditions(product_id):
            return False
            
        # Pobierz dane historyczne
        historical_data = get_historical_data(product_id)
        if historical_data is None or historical_data.empty:
            logger.error(f"Brak danych historycznych dla {product_id}")
            return False
            
        # Pobierz aktualną cenę
        current_price = get_current_price(product_id)
        if not current_price:
            logger.error(f"Nie udało się pobrać aktualnej ceny dla {product_id}")
            return False
            
        try:
            current_price = float(current_price)
        except (ValueError, TypeError) as e:
            logger.error(f"Błąd podczas konwersji aktualnej ceny: {e}")
            return False
            
        # Oblicz wskaźniki techniczne
        prices = historical_data['price'].values
        rsi = calculate_rsi(prices)
        macd, signal, hist = calculate_macd(prices)
        ma_short, ma_long = calculate_moving_averages(prices)
        volatility = calculate_volatility(prices)
        
        # Konwertuj wartości na float
        rsi = float(rsi) if rsi is not None else None
        macd = float(macd) if macd is not None else None
        signal = float(signal) if signal is not None else None
        hist = float(hist) if hist is not None else None
        ma_short = float(ma_short) if ma_short is not None else None
        ma_long = float(ma_long) if ma_long is not None else None
        volatility = float(volatility) if volatility is not None else None
        
        # Sprawdź warunki kupna
        if rsi is not None and rsi < 30:  # Przesprzedany
            logger.info(f"RSI wskazuje na przesprzedanie: {rsi}")
            return True
            
        if macd is not None and signal is not None and ma_short is not None and ma_long is not None:
            if bool(macd > signal and ma_short > ma_long):  # Trend wzrostowy
                logger.info("Wskaźniki wskazują na trend wzrostowy")
                return True
            
        if volatility is not None and volatility < MAX_VOLATILITY * 0.5:  # Niska zmienność
            logger.info(f"Zmienność jest akceptowalna: {volatility}")
            return True
            
        return False
        
    except Exception as e:
        logger.error(f"Błąd podczas sprawdzania warunków kupna: {e}")
        return False

def check_sell_conditions(product_id):
    """Sprawdza warunki sprzedaży dla danej pary."""
    try:
        if product_id not in market_data:
            return True
            
        # Sprawdź warunki handlowe
        if not check_trading_conditions(product_id):
            return False
            
        # Pobierz dane historyczne
        historical_data = get_historical_data(product_id)
        if historical_data is None or historical_data.empty:
            logger.error(f"Brak danych historycznych dla {product_id}")
            return False
            
        # Pobierz aktualną cenę
        current_price = get_current_price(product_id)
        if not current_price:
            logger.error(f"Nie udało się pobrać aktualnej ceny dla {product_id}")
            return False
            
        try:
            current_price = float(current_price)
        except (ValueError, TypeError) as e:
            logger.error(f"Błąd podczas konwersji aktualnej ceny: {e}")
            return False
            
        # Oblicz wskaźniki techniczne
        prices = historical_data['price'].values
        rsi = calculate_rsi(prices)
        macd, signal, hist = calculate_macd(prices)
        ma_short, ma_long = calculate_moving_averages(prices)
        volatility = calculate_volatility(prices)
        
        # Konwertuj wartości na float
        rsi = float(rsi) if rsi is not None else None
        macd = float(macd) if macd is not None else None
        signal = float(signal) if signal is not None else None
        hist = float(hist) if hist is not None else None
        ma_short = float(ma_short) if ma_short is not None else None
        ma_long = float(ma_long) if ma_long is not None else None
        volatility = float(volatility) if volatility is not None else None
        
        # Sprawdź warunki sprzedaży
        if rsi is not None and rsi > 70:  # Przesprzedany
            logger.info(f"RSI zbyt wysoki: {rsi}")
            return True
            
        if macd is not None and signal is not None and ma_short is not None and ma_long is not None:
            if bool(macd < signal and ma_short < ma_long):  # Trend spadkowy
                logger.info("Wskaźniki wskazują na trend spadkowy")
                return True
            
        if volatility is not None and volatility > MAX_VOLATILITY:  # Wysoka zmienność
            logger.info("Zmienność jest zbyt wysoka")
            return True
            
        return False
        
    except Exception as e:
        logger.error(f"Błąd podczas sprawdzania warunków sprzedaży: {e}")
        return False

def check_all_limits(product_id):
    """Sprawdza wszystkie limity handlowe dla danej pary."""
    try:
        # Sprawdź całkowite narażenie
        if not check_total_exposure():
            logger.warning("Przekroczono limit całkowitego narażenia")
            return False
            
        # Sprawdź czy można otworzyć nową pozycję
        if not can_open_new_position():
            logger.warning("Nie można otworzyć nowej pozycji")
            return False
            
        # Sprawdź dzienny P&L
        daily_pnl = check_daily_pnl()
        pnl_limit = calculate_pnl_limit()
        if daily_pnl <= -pnl_limit:
            logger.warning(f"Przekroczono limit dziennego P&L: {daily_pnl} (limit: {pnl_limit})")
            return False
            
        # Sprawdź limit zmienności
        historical_data = get_historical_data(product_id)
        if historical_data is None or historical_data.empty:
            logger.error(f"Brak danych historycznych dla {product_id}")
            return False
            
        volatility = calculate_volatility(historical_data['price'].values)
        volatility_limit = calculate_volatility_limit()
        if volatility > volatility_limit:
            logger.warning(f"Przekroczono limit zmienności: {volatility} (limit: {volatility_limit})")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Błąd podczas sprawdzania limitów: {e}")
        return False

def check_trading_conditions(product_id):
    """Sprawdza warunki handlowe dla danej pary."""
    try:
        if product_id not in market_data:
            return True
            
        # Sprawdź wszystkie limity
        if not check_all_limits(product_id):
            return False
            
        # Sprawdź warunki rynkowe
        if not check_market_conditions(product_id):
            return False
            
        # Pobierz historię transakcji
        trade_history = market_data[product_id]['trade_history']
        if not trade_history:
            return True
            
        # Znajdź ostatnią transakcję
        last_trade = trade_history[-1]
        if not isinstance(last_trade['timestamp'], datetime):
            logger.error("Nieprawidłowy format czasu ostatniej transakcji")
            return False
            
        # Sprawdź czas od ostatniej transakcji
        time_since_last_trade = datetime.utcnow() - last_trade['timestamp']
        if time_since_last_trade.total_seconds() < MIN_TIME_BETWEEN_TRADES:
            logger.warning(f"Zbyt krótki czas od ostatniej transakcji: {time_since_last_trade.total_seconds()}s (minimum: {MIN_TIME_BETWEEN_TRADES}s)")
            return False
            
        # Sprawdź typ ostatniej transakcji
        if last_trade['type'] == 'buy':
            # Jeśli ostatnia transakcja to kupno, sprawdź czy minął wystarczający czas
            if time_since_last_trade.total_seconds() < MIN_HOLDING_TIME:
                logger.warning(f"Zbyt krótki czas trzymania pozycji: {time_since_last_trade.total_seconds()}s (minimum: {MIN_HOLDING_TIME}s)")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Błąd podczas sprawdzania warunków handlowych: {e}")
        return False

def check_daily_pnl():
    """Sprawdza dzienny P&L dla wszystkich par handlowych"""
    try:
        total_pnl = 0
        for pair in TRADING_PAIRS:
            # Pobierz historię transakcji dla pary
            trades = get_transaction_history(pair)
            if not trades:
                continue
                
            # Filtruj transakcje z dzisiejszego dnia
            today = datetime.now().date()
            today_trades = []
            
            for trade in trades:
                try:
                    # Konwertuj timestamp na datetime jeśli jest stringiem
                    if isinstance(trade['timestamp'], str):
                        trade_date = datetime.fromisoformat(trade['timestamp'].replace('Z', '+00:00')).date()
        else:
                        trade_date = trade['timestamp'].date()
                        
                    if trade_date == today:
                        today_trades.append(trade)
                except Exception as e:
                    logger.error(f"Błąd podczas przetwarzania transakcji: {e}")
                    continue
            
            # Oblicz P&L dla dzisiejszych transakcji
            for trade in today_trades:
                amount = float(trade['amount'])
                price = float(trade['price'])
                if trade['type'] == 'BUY':
                    total_pnl -= amount * price
                else:  # SELL
                    total_pnl += amount * price
                    
        return total_pnl
    except Exception as e:
        logger.error(f"Błąd podczas sprawdzania dziennego P&L: {e}")
        return 0

def calculate_pnl_limit():
    """Oblicza limit dziennego P&L."""
    try:
        return MAX_TOTAL_EXPOSURE * MAX_LOSS_PERCENT / 100
    except Exception as e:
        logger.error(f"Błąd podczas obliczania limitu dziennego P&L: {e}")
        return 0

def calculate_volatility_limit():
    """Oblicza limit zmienności."""
    try:
        return MAX_TOTAL_EXPOSURE * MAX_LOSS_PERCENT / 100
    except Exception as e:
        logger.error(f"Błąd podczas obliczania limitu zmienności: {e}")
        return 0

def calculate_rsi_limit():
    """Oblicza limit RSI."""
    try:
        return MAX_TOTAL_EXPOSURE * MAX_LOSS_PERCENT / 100
    except Exception as e:
        logger.error(f"Błąd podczas obliczania limitu RSI: {e}")
        return 0

def calculate_macd_limit():
    """Oblicza limit MACD."""
    try:
        return MAX_TOTAL_EXPOSURE * MAX_LOSS_PERCENT / 100
    except Exception as e:
        logger.error(f"Błąd podczas obliczania limitu MACD: {e}")

def check_stop_loss(product_id):
    """Sprawdza warunki stop loss dla danej pary."""
    try:
        if product_id not in market_data:
            return True
        
        # Pobierz historię transakcji
        trade_history = market_data[product_id]['trade_history']
        if not trade_history:
            return True
            
        # Znajdź ostatnią transakcję kupna
        last_buy = None
        for trade in reversed(trade_history):
            if trade['type'] == 'buy':
                last_buy = trade
                break
                
        if not last_buy:
            return True
                            
        # Pobierz aktualną cenę
        current_price = get_current_price(product_id)
        if not current_price:
            logger.error(f"Nie udało się pobrać aktualnej ceny dla {product_id}")
            return False
            
        try:
            current_price = float(current_price)
            last_buy_price = float(last_buy['price'])
        except (ValueError, TypeError) as e:
            logger.error(f"Błąd podczas konwersji wartości: {e}")
            return False
            
        # Oblicz procentową zmianę ceny
        price_change = (current_price - last_buy_price) / last_buy_price * 100
        
        # Sprawdź warunki stop loss
        if price_change <= -STOP_LOSS_PERCENT:
            logger.warning(f"Stop loss osiągnięty: {price_change}% (limit: -{STOP_LOSS_PERCENT}%)")
            return True
            
        return False
        
    except Exception as e:
        logger.error(f"Błąd podczas sprawdzania warunków stop loss: {e}")
        return False

def check_take_profit(product_id):
    """Sprawdza warunki take profit dla danej pary."""
    try:
        if product_id not in market_data:
            return True
            
        # Pobierz historię transakcji
        trade_history = market_data[product_id]['trade_history']
        if not trade_history:
            return True
            
        # Znajdź ostatnią transakcję kupna
        last_buy = None
        for trade in reversed(trade_history):
            if trade['type'] == 'buy':
                last_buy = trade
                break
                
        if not last_buy:
            return True
            
        # Pobierz aktualną cenę
        current_price = get_current_price(product_id)
        if not current_price:
            logger.error(f"Nie udało się pobrać aktualnej ceny dla {product_id}")
            return False
            
        try:
            current_price = float(current_price)
            last_buy_price = float(last_buy['price'])
        except (ValueError, TypeError) as e:
            logger.error(f"Błąd podczas konwersji wartości: {e}")
            return False
            
        # Oblicz procentową zmianę ceny
        price_change = (current_price - last_buy_price) / last_buy_price * 100
        
        # Sprawdź warunki take profit
        if price_change >= TAKE_PROFIT_PERCENT:
            logger.info(f"Take profit osiągnięty: {price_change}% (limit: {TAKE_PROFIT_PERCENT}%)")
                return True
                
        return False
        
    except Exception as e:
        logger.error(f"Błąd podczas sprawdzania warunków take profit: {e}")
        return False

def check_trailing_stop(product_id):
    """Sprawdza warunki trailing stop dla danej pary."""
    try:
        if product_id not in market_data:
            return True
            
        # Pobierz historię transakcji
        trade_history = market_data[product_id]['trade_history']
        if not trade_history:
            return True
            
        # Znajdź ostatnią transakcję kupna
        last_buy = None
        for trade in reversed(trade_history):
            if trade['type'] == 'buy':
                last_buy = trade
                break
                                    
        if not last_buy:
            return True
            
        # Pobierz aktualną cenę
        current_price = get_current_price(product_id)
        if not current_price:
            logger.error(f"Nie udało się pobrać aktualnej ceny dla {product_id}")
            return False
            
        try:
            current_price = float(current_price)
            last_buy_price = float(last_buy['price'])
        except (ValueError, TypeError) as e:
            logger.error(f"Błąd podczas konwersji wartości: {e}")
            return False
            
        # Oblicz procentową zmianę ceny
        price_change = (current_price - last_buy_price) / last_buy_price * 100
        
        # Sprawdź warunki trailing stop
        if price_change <= -TRAILING_STOP_PERCENT:
            logger.warning(f"Trailing stop osiągnięty: {price_change}% (limit: -{TRAILING_STOP_PERCENT}%)")
            return True
            
        return False
                
    except Exception as e:
        logger.error(f"Błąd podczas sprawdzania warunków trailing stop: {e}")
        return False

def check_all_conditions(product_id):
    """Sprawdza wszystkie warunki dla danej pary."""
    try:
        if product_id not in market_data:
            return True
            
        # Sprawdź warunki handlowe
        if not check_trading_conditions(product_id):
            return False
            
        # Sprawdź warunki rynkowe
        if not check_market_conditions(product_id):
            return False
            
        # Sprawdź warunki kupna
        if not check_buy_conditions(product_id):
            return False
            
        # Sprawdź warunki sprzedaży
        if not check_sell_conditions(product_id):
            return False
            
        # Sprawdź warunki stop loss
        if not check_stop_loss(product_id):
            return False
            
        # Sprawdź warunki take profit
        if not check_take_profit(product_id):
            return False
            
        # Sprawdź warunki trailing stop
        if not check_trailing_stop(product_id):
            return False
            
        return True
                
    except Exception as e:
        logger.error(f"Błąd podczas sprawdzania wszystkich warunków: {e}")
        return False

def check_market_conditions(product_id):
    """Sprawdza warunki rynkowe dla danej pary."""
    try:
        global MAX_VOLATILITY
        if product_id not in market_data:
            return True
            
        # Pobierz dane historyczne
        historical_data = get_historical_data(product_id)
        if historical_data is None or historical_data.empty:
            logger.error(f"Brak danych historycznych dla {product_id}")
            return False
            
        # Oblicz wskaźniki techniczne
        prices = historical_data['price'].values
        rsi = calculate_rsi(prices)
        macd, signal, hist = calculate_macd(prices)
        ma_short, ma_long = calculate_moving_averages(prices)
        volatility = calculate_volatility(prices)
        
        # Sprawdź czy wszystkie wskaźniki są dostępne
        if rsi is None or macd is None or signal is None or ma_short is None or ma_long is None:
            return False
            
        # Sprawdź warunki rynkowe
        if rsi > 70 or rsi < 30:  # Ekstremalne wartości RSI
            return False
            
        if volatility > MAX_VOLATILITY:  # Zbyt wysoka zmienność
            return False
            
        if macd * signal < 0:  # Konflikt między MACD a sygnałem
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Błąd podczas sprawdzania warunków rynkowych: {e}")
        return False

def get_current_price(product_id):
    """Pobiera aktualną cenę dla danej pary."""
    try:
        if product_id not in market_data:
            return None
            
        current_price = market_data[product_id].get('current_price')
        if current_price is None:
            # Spróbuj pobrać cenę z API
            ticker = get_product_ticker(product_id)
            if ticker:
                market_data[product_id]['current_price'] = ticker
                return ticker
            return None
            
        return current_price
        
    except Exception as e:
        logger.error(f"Błąd podczas pobierania aktualnej ceny: {e}")
        return None

def get_transaction_history(pair):
    """Pobiera historię transakcji dla danej pary."""
    try:
        if pair not in market_data:
            return []
            
        return market_data[pair].get('trade_history', [])
        
    except Exception as e:
        logger.error(f"Błąd podczas pobierania historii transakcji: {e}")
        return []

def calculate_market_sentiment(pair):
    """Oblicza sentyment rynkowy dla danej pary."""
    try:
        if pair not in market_data:
            return 0
                        
                        # Pobierz dane historyczne
                        historical_data = get_historical_data(pair)
        if historical_data is None or historical_data.empty:
            return 0
                        
                        # Oblicz wskaźniki techniczne
        prices = historical_data['price'].values
        rsi = calculate_rsi(prices)
        macd, signal, hist = calculate_macd(prices)
        ma_short, ma_long = calculate_moving_averages(prices)
        
        if rsi is None or macd is None or signal is None or ma_short is None or ma_long is None:
            return 0
            
        # Oblicz sentyment na podstawie wskaźników
        sentiment = 0
        
        # RSI
        if rsi > 70:
            sentiment -= 1
        elif rsi < 30:
            sentiment += 1
            
        # MACD
        if macd > signal:
            sentiment += 1
        else:
            sentiment -= 1
            
        # Średnie kroczące
        if ma_short > ma_long:
            sentiment += 1
        else:
            sentiment -= 1
            
        return sentiment / 3  # Normalizacja do zakresu [-1, 1]
                
            except Exception as e:
        logger.error(f"Błąd podczas obliczania sentymentu rynkowego: {e}")
        return 0

def main():
    logger.info('Uruchamianie bota...')
    try:
        ws.connect()
        while True:
            for pair in TRADING_PAIRS:
                if check_all_conditions(pair):
                    logger.info(f'Warunki handlowe spełnione dla {pair}')
            time.sleep(TRADE_INTERVAL)
    except Exception as e:
        logger.error(f'Krytyczny błąd: {e}')
    finally:
        ws.disconnect()
        
if __name__ == "__main__":
    try:
    main()
    except Exception as e:
        logger.error(f"Krytyczny błąd: {str(e)}")
        raise
