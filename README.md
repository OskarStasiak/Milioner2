# Crypto Trading Bot

Bot handlowy do automatycznego handlu kryptowalutami na giełdzie Coinbase.

## Funkcje

- Handel na parach ETH-USDC i BTC-USDC
- Analiza techniczna (RSI, MACD, średnie kroczące)
- Automatyczne wykonywanie zleceń kupna/sprzedaży
- Monitorowanie sald i historii transakcji
- Logowanie aktywności

## Wymagania

- Python 3.8+
- Biblioteki: coinbase, pandas, numpy, tensorflow, websocket-client

## Instalacja

1. Sklonuj repozytorium:
```bash
git clone https://github.com/twoj-username/crypto-trading-bot.git
cd crypto-trading-bot
```

2. Zainstaluj wymagane biblioteki:
```bash
pip install -r requirements.txt
```

3. Utwórz plik `cdp_api_key.json` z kluczami API:
```json
{
    "api_key_id": "twój_api_key_id",
    "api_key_secret": "twój_api_key_secret"
}
```

4. Utwórz plik `production.env` z parametrami handlowymi:
```
TRADING_PAIRS=ETH-USDC,BTC-USDC
TRADE_VALUE_USDC=50
MAX_TOTAL_EXPOSURE=300
MAX_POSITION_SIZE=100
```

## Uruchomienie

```bash
python3 crypto_bot.py
```

## Bezpieczeństwo

- Nigdy nie udostępniaj plików `cdp_api_key.json` i `production.env`
- Używaj silnych kluczy API z odpowiednimi uprawnieniami
- Regularnie monitoruj aktywność bota

## Licencja

MIT 