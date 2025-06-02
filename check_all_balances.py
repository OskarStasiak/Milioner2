import os
from dotenv import load_dotenv
from coinbase.rest import RESTClient
from pathlib import Path
import requests

# Załaduj zmienne środowiskowe
load_dotenv(str(Path(__file__).parent / 'production.env'))

API_KEY = os.getenv('CDP_API_KEY_ID')
API_SECRET = os.getenv('CDP_API_KEY_SECRET')
client = RESTClient(api_key=API_KEY, api_secret=API_SECRET)

def get_balance_value(balance_obj):
    if hasattr(balance_obj, 'value'):
        return float(balance_obj.value)
    elif isinstance(balance_obj, dict) and 'value' in balance_obj:
        return float(balance_obj['value'])
    return 0.0

def get_prices(symbols):
    # Mapowanie symboli na id CoinGecko
    coingecko_map = {
        'BTC': 'bitcoin', 'ETH': 'ethereum', 'USDC': 'usd-coin', 'USDT': 'tether',
        'SOL': 'solana', 'DOGE': 'dogecoin', 'XRP': 'ripple', 'PEPE': 'pepe',
        # Dodaj inne jeśli potrzeba
    }
    ids = ','.join([coingecko_map.get(sym, '').lower() for sym in symbols if sym in coingecko_map])
    if not ids:
        return {}
    url = f'https://api.coingecko.com/api/v3/simple/price?ids={ids}&vs_currencies=usd'
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        return {sym: data[coingecko_map[sym]]['usd'] for sym in symbols if sym in coingecko_map and coingecko_map[sym] in data}
    except Exception as e:
        print(f"Nie można pobrać kursów: {e}")
        return {}

def main():
    print("\n" + "="*70)
    print("DOSTĘPNE ŚRODKI NA WSZYSTKICH WALUTACH".center(70))
    print("="*70)
    try:
        accounts = client.get_accounts().accounts
        symbols = [a.currency for a in accounts if hasattr(a, 'currency')]
        prices = get_prices(symbols)
        print(f"{'Waluta':<8} {'Dostępne':>15} {'Całkowite':>15} {'W USD':>18}")
        print("-"*70)
        for acc in accounts:
            if hasattr(acc, 'currency'):
                symbol = acc.currency
                available = get_balance_value(acc.available_balance)
                hold = get_balance_value(acc.hold)
                total = available + hold
                if available < 1e-8 and total < 1e-8:
                    continue  # pomiń puste
                usd_val = prices.get(symbol, None)
                usd_str = f"${available * usd_val:.2f}" if usd_val else "-"
                print(f"{symbol:<8} {available:>15.8f} {total:>15.8f} {usd_str:>18}")
    except Exception as e:
        print(f"Błąd podczas pobierania sald: {e}")

if __name__ == "__main__":
    main() 