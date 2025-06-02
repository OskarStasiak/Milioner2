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
from crypto_bot import check_all_conditions, check_all_limits

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
TRADING_PAIRS = ['ETH-USDC', 'BTC-USDC', 'SOL-USDC', 'DOGE-USDC', 'XRP-USDC', 'MATIC-USDC', 'LINK-USDC']
TRADE_VALUE_USDC = float(os.getenv('TRADE_VALUE_USDC', 50))
MAX_TOTAL_EXPOSURE = float(os.getenv('MAX_TOTAL_EXPOSURE', 200))
MAX_POSITION_SIZE = float(os.getenv('MAX_POSITION_SIZE', 0.05))
PRICE_THRESHOLD_BUY = float(os.getenv('PRICE_THRESHOLD_BUY', 2500))
PRICE_THRESHOLD_SELL = float(os.getenv('PRICE_THRESHOLD_SELL', 2800))
MIN_PROFIT_PERCENT = float(os.getenv('MIN_PROFIT_PERCENT', 0.3))
MAX_LOSS_PERCENT = float(os.getenv('MAX_LOSS_PERCENT', 1.0))
STOP_LOSS_PERCENT = float(os.getenv('STOP_LOSS_PERCENT', 0.8))
TAKE_PROFIT_PERCENT = float(os.getenv('TAKE_PROFIT_PERCENT', 1.0))
TRAILING_STOP_PERCENT = float(os.getenv('TRAILING_STOP_PERCENT', 0.3))
TRADE_INTERVAL = int(os.getenv('TRADE_INTERVAL', 30))

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
                if len(market_data[product_id]['price_history']) > 1000:
                    market_data[product_id]['price_history'].pop(0)
                logging.info(f"Aktualna cena {product_id}: {price}")
            
        elif data['type'] == 'snapshot':
            product_id = data.get('product_id')
            if product_id not in TRADING_PAIRS:
                return
                
            market_data[product_id]['order_book']['bids'] = data.get('bids', [])
            market_data[product_id]['order_book']['asks'] = data.get('asks', [])
            logging.info(f"Zaktualizowano książkę zleceń dla {product_id}")
            
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
            
            market_data[product_id]['order_book']['bids'].sort(key=lambda x: float(x[0]), reverse=True)
            market_data[product_id]['order_book']['asks'].sort(key=lambda x: float(x[0]))
            
    except Exception as e:
        logging.error(f"Błąd podczas przetwarzania wiadomości WebSocket: {e}")

def should_trade(current_price, historical_data):
    """Sprawdza czy należy wykonać transakcję na podstawie wskaźników technicznych."""
    try:
        df = pd.DataFrame(historical_data)
        rsi = calculate_rsi(df['price'].values)
        macd, macd_signal, macd_hist = calculate_macd(df['price'].values)
        ma_short, ma_long = calculate_moving_averages(df['price'].values)
        
        if rsi is None or macd is None or macd_signal is None or ma_short is None or ma_long is None:
            return False
            
        last_price = float(df['price'].iloc[-1])
        prev_price = float(df['price'].iloc[-2])
        
        if last_price > prev_price:
            if 30 <= rsi <= 70:
                if macd > 0:
                    if macd > macd_signal:
                        if (macd - macd_signal) > 0:
                            return True
        return False
        
    except Exception as e:
        logger.error(f"Błąd podczas analizy warunków handlowych: {e}")
        return False

def analyze_market_trend(historical_data):
    """Analizuj trend rynkowy."""
    try:
        if historical_data is None or (isinstance(historical_data, pd.DataFrame) and historical_data.empty) or len(historical_data) < 24:
            return "neutral"
    
        if isinstance(historical_data, pd.DataFrame):
            prices = historical_data['price'].values
        else:
            prices = np.array(historical_data)
            
        if len(prices) == 0:
            return "neutral"
            
        current_price = float(prices[-1])
        ma_short, ma_long = calculate_moving_averages(prices)
        rsi = calculate_rsi(prices)
        
        if ma_short is None or ma_long is None or rsi is None:
            return "neutral"
        
        ma_short = float(ma_short)
        ma_long = float(ma_long)
        rsi = float(rsi)
        
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
                
            last_buy = None
            for trade in reversed(trade_history):
                if trade['type'] == 'buy':
                    last_buy = trade
                    break
                    
            if last_buy:
                position_value = float(last_buy['amount']) * float(last_buy['price'])
                total_exposure += position_value
                    
        return total_exposure
        
    except Exception as e:
        logger.error(f"Błąd podczas sprawdzania całkowitej ekspozycji: {e}")
        return 0

def check_trailing_stop(product_id):
    """Sprawdza warunki trailing stop dla danej pary."""
    try:
        if product_id not in market_data:
            return True
            
        trade_history = market_data[product_id]['trade_history']
        if not trade_history:
            return True
            
        last_buy = None
        for trade in reversed(trade_history):
            if trade['type'] == 'buy':
                last_buy = trade
                break
                                    
        if not last_buy:
            return True
            
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
            
        price_change = (current_price - last_buy_price) / last_buy_price * 100
        
        if price_change <= -TRAILING_STOP_PERCENT:
            logger.warning(f"Trailing stop osiągnięty: {price_change}% (limit: -{TRAILING_STOP_PERCENT}%)")
            return True
            
        return False
                
    except Exception as e:
        logger.error(f"Błąd podczas sprawdzania warunków trailing stop: {e}")
        return False

def check_all_limits(product_id):
    """Tymczasowa funkcja: zawsze zwraca True."""
    return True

def main():
    """Główna funkcja bota."""
    try:
        logger.info("Uruchamianie bota...")
        ws_client = CoinbaseWebSocket(API_KEY, API_SECRET, TRADING_PAIRS)
        ws_client.connect()
        
        while True:
            try:
                for product_id in TRADING_PAIRS:
                    if check_all_conditions(product_id):
                        logger.info(f"Warunki spełnione dla {product_id}")
                
                time.sleep(TRADE_INTERVAL)
                
            except Exception as e:
                logger.error(f"Błąd w głównej pętli: {e}")
                time.sleep(TRADE_INTERVAL)
                
    except KeyboardInterrupt:
        logger.info("Zatrzymywanie bota...")
        ws_client.disconnect()
    except Exception as e:
        logger.error(f"Krytyczny błąd: {e}")
        ws_client.disconnect()

if __name__ == "__main__":
    main() 