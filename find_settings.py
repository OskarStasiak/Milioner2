from datetime import datetime, timedelta
from crypto_bot import load_settings_from_time

# Ustaw czas na wczoraj 7:00
yesterday = datetime.now() - timedelta(days=1)
target_time = yesterday.replace(hour=7, minute=0, second=0, microsecond=0)

# Wczytaj ustawienia
settings = load_settings_from_time(target_time)

if settings:
    print("\nZnalezione ustawienia:")
    print(f"Czas zapisu: {settings['timestamp']}")
    print(f"Symbol: {settings['SYMBOL']}")
    print(f"Wartość transakcji (USDC): {settings['TRADE_VALUE_USDC']}")
    print(f"Próg ceny kupna: {settings['PRICE_THRESHOLD_BUY']}")
    print(f"Próg ceny sprzedaży: {settings['PRICE_THRESHOLD_SELL']}")
    print(f"Procent stop loss: {settings['STOP_LOSS_PERCENTAGE']}")
    print(f"Maksymalny procent straty: {settings['MAX_LOSS_PERCENTAGE']}")
else:
    print("Nie znaleziono ustawień z tego okresu") 