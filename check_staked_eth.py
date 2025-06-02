import os
from dotenv import load_dotenv
from coinbase.rest import RESTClient
from pathlib import Path
import json
from datetime import datetime, timedelta
import requests

# Załaduj zmienne środowiskowe
load_dotenv(str(Path(__file__).parent / 'production.env'))

# Konfiguracja API
API_KEY = os.getenv('CDP_API_KEY_ID')
API_SECRET = os.getenv('CDP_API_KEY_SECRET')

# Inicjalizacja klienta
client = RESTClient(api_key=API_KEY, api_secret=API_SECRET)

def format_balance(balance):
    """Formatuje saldo do czytelnej postaci."""
    return f"{float(balance):.8f}"

def get_balance_value(balance_obj):
    """Bezpiecznie pobiera wartość salda z obiektu."""
    if hasattr(balance_obj, 'value'):
        return float(balance_obj.value)
    elif isinstance(balance_obj, dict) and 'value' in balance_obj:
        return float(balance_obj['value'])
    return 0.0

def get_eth_usd_price():
    """Pobiera aktualny kurs ETH/USD z CoinGecko."""
    try:
        url = 'https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd'
        response = requests.get(url, timeout=10)
        data = response.json()
        return float(data['ethereum']['usd'])
    except Exception as e:
        print(f"Nie można pobrać kursu ETH/USD: {e}")
        return None

def check_staked_eth():
    """Sprawdza status zablokowanego ETH i historię stakingu."""
    try:
        print("\n" + "="*50)
        print("STATUS ZABLOKOWANEGO ETH".center(50))
        print("="*50)
        
        eth_usd = get_eth_usd_price()
        
        accounts = client.get_accounts().accounts
        
        for account in accounts:
            if hasattr(account, 'currency') and account.currency == 'ETH':
                print(f"\n📊 KONTO ETH")
                print("-"*50)
                print(f"ID: {account.uuid}")
                print(f"Status: {'✅ Aktywne' if account.active else '❌ Nieaktywne'}")
                
                # Sprawdź zablokowane środki
                if hasattr(account, 'hold'):
                    hold = get_balance_value(account.hold)
                    available = get_balance_value(account.available_balance)
                    total = available + hold
                    
                    print("\n💰 SALDA")
                    print("-"*50)
                    print(f"Dostępne:     {format_balance(available)} ETH", end='')
                    if eth_usd:
                        print(f"   ≈ ${available * eth_usd:.2f} USD")
                    else:
                        print()
                    print(f"Zablokowane:  {format_balance(hold)} ETH", end='')
                    if eth_usd:
                        print(f"   ≈ ${hold * eth_usd:.2f} USD")
                    else:
                        print()
                    print(f"Całkowite:    {format_balance(total)} ETH", end='')
                    if eth_usd:
                        print(f"   ≈ ${total * eth_usd:.2f} USD")
                    else:
                        print()
                    
                    # Oblicz szacowany czas odblokowania (zakładając 24-48h)
                    print("\n⏳ STATUS ODBLOKOWANIA")
                    print("-"*50)
                    print("Środki są w okresie 'unbonding' (odblokowywania)")
                    print("Typowy czas odblokowania: 24-48 godzin")
                    print("Sprawdź ponownie za kilka godzin")
                    
                    # Dodaj informację o wartości w USD
                    try:
                        ticker = client.get_product_ticker("ETH-USD")
                        if ticker and hasattr(ticker, 'price'):
                            eth_price = float(ticker.price)
                            hold_value_usd = hold * eth_price
                            print(f"\n💵 WARTOŚĆ ZABLOKOWANYCH ŚRODKÓW")
                            print("-"*50)
                            print(f"Wartość w USD: ${hold_value_usd:.2f}")
                    except Exception as e:
                        print(f"\nNie można pobrać aktualnej ceny ETH: {e}")
                
    except Exception as e:
        print(f"\n❌ Błąd podczas sprawdzania statusu ETH: {e}")

if __name__ == "__main__":
    check_staked_eth() 