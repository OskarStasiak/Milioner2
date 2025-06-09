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

# Wyczyść plik logów przy starcie
with open('crypto_bot.log', 'w') as f:
    f.write('=== NOWA SESJA BOTA ===\n')

# Załaduj zmienne środowiskowe
load_dotenv('production.env')

# Konfiguracja API
with open('cdp_api_key.json', 'r') as f:
    api_keys = json.load(f)
    API_KEY = api_keys['api_key_id']
    API_SECRET = api_keys['api_key_secret']

# Parametry handlowe
TRADING_PAIRS = ['ETH-USD', 'BTC-USD', 'SOL-USD']
TRADE_VALUE_USDC = float(os.getenv('TRADE_VALUE_USDC', 100))
MAX_TOTAL_EXPOSURE = float(os.getenv('MAX_TOTAL_EXPOSURE', 500))
MAX_POSITION_SIZE = float(os.getenv('MAX_POSITION_SIZE', 0.05))
MIN_PROFIT_PERCENT = float(os.getenv('MIN_PROFIT_PERCENT', 0.15))
MAX_LOSS_PERCENT = float(os.getenv('MAX_LOSS_PERCENT', 1.2))
STOP_LOSS_PERCENT = float(os.getenv('STOP_LOSS_PERCENT', 0.8))
TAKE_PROFIT_PERCENT = float(os.getenv('TAKE_PROFIT_PERCENT', 0.6))
TRAILING_STOP_PERCENT = float(os.getenv('TRAILING_STOP_PERCENT', 0.4))
TRADE_INTERVAL = int(os.getenv('TRADE_INTERVAL', 30))

# Inicjalizacja API
client = RESTClient(api_key=API_KEY, api_secret=API_SECRET)

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
                logger.info(f"📊 Otrzymano nową cenę dla {product_id}: {price}")
                logger.info(f"📈 Historia cen dla {product_id}: {len(market_data[product_id]['price_history'])} punktów")
            
        elif data['type'] == 'snapshot':
            product_id = data.get('product_id')
            if product_id not in TRADING_PAIRS:
                return
                
            market_data[product_id]['order_book']['bids'] = data.get('bids', [])
            market_data[product_id]['order_book']['asks'] = data.get('asks', [])
            logger.info(f"📚 Zaktualizowano książkę zleceń dla {product_id}")
            logger.info(f"📊 Liczba bidów: {len(market_data[product_id]['order_book']['bids'])}")
            logger.info(f"📊 Liczba asków: {len(market_data[product_id]['order_book']['asks'])}")
            
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
            logger.info(f"📈 Zaktualizowano L2 dla {product_id}")
            
    except Exception as e:
        logger.error(f"❌ Błąd podczas przetwarzania wiadomości WebSocket: {e}")

# Inicjalizacja WebSocket dla wszystkich par
ws = CoinbaseWebSocket(API_KEY, API_SECRET, TRADING_PAIRS, callback=on_ws_message)

def calculate_rsi(prices, period=14):
    """Oblicza wskaźnik RSI."""
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum()/period
    down = -seed[seed < 0].sum()/period
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:period] = 100. - 100./(1.+rs)

    for i in range(period, len(prices)):
        delta = deltas[i-1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(period-1) + upval)/period
        down = (down*(period-1) + downval)/period
        rs = up/down
        rsi[i] = 100. - 100./(1.+rs)

    return rsi

def should_trade(current_price, historical_data):
    """Ulepszona strategia handlowa z lepszą analizą trendu."""
    try:
        logger.info("\n=== WEJŚCIE DO FUNKCJI SHOULD_TRADE ===")
        
        if not isinstance(historical_data, (list, pd.DataFrame)):
            logger.error("❌ Nieprawidłowy format danych historycznych")
            return False
            
        if isinstance(historical_data, list):
            if not all(isinstance(x, dict) and 'price' in x for x in historical_data):
                logger.error("❌ Nieprawidłowy format danych cenowych")
                return False
            df = pd.DataFrame(historical_data)
        else:
            df = historical_data
            
        if 'price' not in df.columns:
            logger.error("❌ Brak kolumny 'price' w danych")
            return False
            
        logger.info(f"Liczba punktów danych: {len(df)}")
        
        if len(df) < 12:  # Zwiększamy wymaganą liczbę punktów
            logger.info("❌ Za mało danych historycznych (minimum 12 punktów)")
            return False

        try:
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            if df['price'].isnull().any():
                logger.error("❌ Wykryto nieprawidłowe wartości cenowe")
                return False
                
            last_price = float(df['price'].iloc[-1])
            prev_price = float(df['price'].iloc[-2])
            
            if last_price <= 0 or prev_price <= 0:
                logger.error("❌ Wykryto nieprawidłowe wartości cenowe (cena <= 0)")
                return False
                
            price_change = ((last_price - prev_price) / prev_price) * 100
            
            # Oblicz średnie kroczące
            try:
                ma8 = df['price'].rolling(window=8).mean().iloc[-1]
                ma20 = df['price'].rolling(window=20).mean().iloc[-1]
                ma50 = df['price'].rolling(window=50).mean().iloc[-1]
                if pd.isna(ma8) or pd.isna(ma20) or pd.isna(ma50):
                    logger.error("❌ Nie można obliczyć średnich kroczących")
                    return False
            except Exception as e:
                logger.error(f"❌ Błąd podczas obliczania średnich kroczących: {e}")
                return False
            
            # Analiza wolumenu
            volume_data = False
            if 'volume' in df.columns:
                try:
                    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
                    if df['volume'].isnull().any():
                        logger.warning("⚠️ Wykryto nieprawidłowe wartości wolumenu")
                        volume_increase = True
                    else:
                        volume = float(df['volume'].iloc[-1])
                        avg_volume = float(df['volume'].rolling(window=12).mean().iloc[-1])
                        volume_increase = volume > avg_volume * 1.15  # Zwiększamy wymagany wzrost wolumenu
                        volume_data = True
                        logger.info(f"Wolumen: {volume:.2f} (średnia: {avg_volume:.2f})")
                except Exception as e:
                    logger.warning(f"⚠️ Błąd podczas analizy wolumenu: {e}")
                    volume_increase = True
            else:
                volume_increase = True
                logger.info("ℹ️ Brak danych o wolumenie")
            
            logger.info(f"\nWskaźniki cenowe:")
            logger.info(f"Aktualna cena: {last_price:.2f}")
            logger.info(f"Poprzednia cena: {prev_price:.2f}")
            logger.info(f"Zmiana ceny: {price_change:.2f}%")
            logger.info(f"MA8: {ma8:.2f}")
            logger.info(f"MA20: {ma20:.2f}")
            logger.info(f"MA50: {ma50:.2f}")
            
            # Ulepszone warunki handlowe
            price_above_ma = last_price > ma8 * 0.995  # Zwiększamy tolerancję
            price_momentum = price_change > 0.02  # Zmniejszamy wymagany momentum
            ma_trend = ma8 > ma20 and ma20 > ma50  # Sprawdzamy trend na wszystkich średnich
            volatility_ok = abs(price_change) < 2.0  # Sprawdzamy zmienność
            
            logger.info(f"\nWarunki handlowe:")
            logger.info(f"Cena powyżej MA8: {'✅' if price_above_ma else '❌'}")
            logger.info(f"Pozytywny momentum: {'✅' if price_momentum else '❌'}")
            logger.info(f"Trend MA8 > MA20 > MA50: {'✅' if ma_trend else '❌'}")
            logger.info(f"Zmienność OK: {'✅' if volatility_ok else '❌'}")
            if volume_data:
                logger.info(f"Wzrost wolumenu: {'✅' if volume_increase else '❌'}")
            
            # Sygnał do handlu - wymagamy 3 z 4 warunków
            conditions_met = sum([price_above_ma, price_momentum, ma_trend, volatility_ok])
            if conditions_met >= 3:
                logger.info("\n🎯 SYGNAŁ DO HANDLU - spełnione warunki:")
                if price_above_ma:
                    logger.info(f"- Cena ({last_price:.2f}) > MA8 ({ma8:.2f})")
                if price_momentum:
                    logger.info(f"- Momentum: {price_change:.2f}%")
                if ma_trend:
                    logger.info(f"- Trend MA8 > MA20 > MA50")
                if volatility_ok:
                    logger.info(f"- Zmienność OK: {abs(price_change):.2f}%")
                return True
            else:
                logger.info("\n❌ BRAK SYGNAŁU DO HANDLU")
                if not price_above_ma:
                    logger.info(f"- Cena ({last_price:.2f}) poniżej MA8 ({ma8:.2f})")
                if not price_momentum:
                    logger.info(f"- Momentum ({price_change:.2f}%) zbyt słaby")
                if not ma_trend:
                    logger.info(f"- Trend nie potwierdzony")
                if not volatility_ok:
                    logger.info(f"- Zmienność zbyt wysoka: {abs(price_change):.2f}%")
                return False
                
        except Exception as e:
            logger.error(f"❌ Błąd podczas analizy danych: {e}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Krytyczny błąd w funkcji should_trade: {e}")
        return False

def analyze_market_trend(historical_data):
    """Analizuj trend rynkowy."""
    try:
        if historical_data is None or (isinstance(historical_data, pd.DataFrame) and historical_data.empty) or len(historical_data) < 24:
            logger.warning("Niewystarczająca ilość danych do analizy trendu")
            return "neutral"
    
        if isinstance(historical_data, pd.DataFrame):
            prices = historical_data['price'].values
        else:
            prices = np.array(historical_data)
            
        if len(prices) == 0:
            logger.warning("Brak danych cenowych")
            return "neutral"
            
        current_price = float(prices[-1])
        ma_short, ma_long = calculate_moving_averages(prices)
        rsi = calculate_rsi(prices)
        
        logger.info(f"Analiza trendu:")
        logger.info(f"Aktualna cena: {current_price}")
        logger.info(f"MA Short: {ma_short}, MA Long: {ma_long}")
        logger.info(f"RSI: {rsi[-1]:.2f}")
        
        if ma_short is None or ma_long is None or rsi is None:
            logger.warning("Brak wystarczających danych do analizy trendu")
            return "neutral"
        
        ma_short = float(ma_short)
        ma_long = float(ma_long)
        rsi = float(rsi[-1])
        
        trend_bullish = current_price > ma_short and ma_short > ma_long and rsi > 50
        trend_bearish = current_price < ma_short and ma_short < ma_long and rsi < 50
        
        if trend_bullish:
            logger.info("Wykryto trend wzrostowy")
            return "bullish"
        elif trend_bearish:
            logger.info("Wykryto trend spadkowy")
            return "bearish"
        else:
            logger.info("Wykryto trend neutralny")
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

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Oblicza wskaźnik MACD."""
    try:
        exp1 = pd.Series(prices).ewm(span=fast, adjust=False).mean()
        exp2 = pd.Series(prices).ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return macd.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1]
    except Exception as e:
        logger.error(f"Błąd podczas obliczania MACD: {e}")
        return None, None, None

def calculate_moving_averages(prices, short_period=20, long_period=50):
    """Oblicza średnie kroczące."""
    try:
        short_ma = pd.Series(prices).rolling(window=short_period).mean()
        long_ma = pd.Series(prices).rolling(window=long_period).mean()
        return short_ma.iloc[-1], long_ma.iloc[-1]
    except Exception as e:
        logger.error(f"Błąd podczas obliczania średnich kroczących: {e}")
        return None, None

def main():
    """Główna funkcja bota."""
    try:
        logger.info("🚀 Uruchamianie bota...")
        
        # Inicjalizacja danych dla każdej pary
        for pair in TRADING_PAIRS:
            logger.info(f"📊 Inicjalizacja danych dla {pair}")
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
        
        # Sprawdzenie połączenia z API
        try:
            ws_client = CoinbaseWebSocket(API_KEY, API_SECRET, TRADING_PAIRS)
            ws_client.connect()
            logger.info("✅ Połączenie z WebSocket nawiązane")
        except Exception as e:
            logger.error(f"❌ Błąd podczas łączenia z WebSocket: {e}")
            return
        
        while True:
            try:
                for product_id in TRADING_PAIRS:
                    logger.info(f"\n{'='*50}")
                    logger.info(f"🔍 ANALIZA {product_id}")
                    logger.info(f"{'='*50}")
                    
                    # Sprawdzenie dostępności danych
                    if product_id not in market_data:
                        logger.error(f"❌ Brak danych dla {product_id}")
                        continue
                        
                    if market_data[product_id]['current_price'] is None:
                        logger.info(f"⏳ Oczekiwanie na dane dla {product_id}")
                        continue
                        
                    current_price = market_data[product_id]['current_price']
                    historical_data = market_data[product_id]['price_history']
                    
                    logger.info(f"📊 Stan danych:")
                    logger.info(f"- Aktualna cena: {current_price}")
                    logger.info(f"- Liczba punktów w historii: {len(historical_data)}")
                    if len(historical_data) > 0:
                        logger.info(f"- Ostatnia cena w historii: {historical_data[-1]['price']}")
                        logger.info(f"- Timestamp ostatniej ceny: {historical_data[-1]['timestamp']}")
                    
                    # Walidacja danych historycznych
                    if not isinstance(historical_data, list):
                        logger.error(f"❌ Nieprawidłowy format danych historycznych dla {product_id}")
                        continue
                        
                    if len(historical_data) >= 20:
                        try:
                            logger.info(f"📈 Rozpoczynam analizę warunków handlowych...")
                            should_trade_result = should_trade(current_price, historical_data)
                            if should_trade_result:
                                logger.info(f"🎯 Wykryto sygnał do handlu dla {product_id}!")
                                # TODO: Implementacja wykonania zlecenia
                            else:
                                logger.info(f"⏳ Oczekiwanie na lepsze warunki dla {product_id}")
                        except Exception as e:
                            logger.error(f"❌ Błąd podczas analizy {product_id}: {e}")
                    else:
                        logger.info(f"📊 Zbieranie danych dla {product_id}... ({len(historical_data)}/20)")
                
                logger.info(f"\n⏳ Oczekiwanie {TRADE_INTERVAL} sekund przed następną analizą...")
                time.sleep(TRADE_INTERVAL)
                
            except Exception as e:
                logger.error(f"❌ Błąd w głównej pętli: {e}")
                time.sleep(TRADE_INTERVAL)
                
    except KeyboardInterrupt:
        logger.info("🛑 Zatrzymywanie bota...")
        try:
            ws_client.disconnect()
        except:
            pass
    except Exception as e:
        logger.error(f"❌ Krytyczny błąd: {e}")
        try:
            ws_client.disconnect()
        except:
            pass

if __name__ == "__main__":
    main() 