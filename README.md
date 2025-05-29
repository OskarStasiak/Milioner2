# Crypto Trading Bot

Zaawansowany bot do handlu kryptowalutami na giełdzie Coinbase, wykorzystujący analizę techniczną i dynamiczne zarządzanie zyskami.

## Funkcje

- Handel na parach ETH-USDC i BTC-USDC
- Dynamiczne zarządzanie zyskami oparte na analizie rynku
- Automatyczne stop-loss i take-profit
- Analiza techniczna (RSI, MACD, średnie kroczące)
- Zarządzanie ryzykiem i kapitałem
- Monitoring w czasie rzeczywistym

## Parametry handlowe

- Minimalny cel zysku: 0.5%
- Maksymalny cel zysku: 5%
- Stop-loss: 2%
- Trailing stop: 1%
- Maksymalna liczba transakcji dziennie: 5
- Minimalna wielkość zlecenia: 20 USDC

## Wymagania

- Python 3.8+
- Konto na Coinbase z API
- Wymagane pakiety: pandas, numpy, tensorflow, coinbase

## Instalacja

1. Sklonuj repozytorium:
```bash
git clone https://github.com/twoj-username/crypto-trading-bot.git
cd crypto-trading-bot
```

2. Zainstaluj wymagane pakiety:
```bash
pip install -r requirements.txt
```

3. Skonfiguruj plik .env z kluczami API:
```
API_KEY=twoj_klucz_api
API_SECRET=twoj_sekret_api
```

## Użycie

Uruchom bota:
```bash
python3 crypto_bot.py
```

Sprawdź salda:
```bash
python3 check_balance.py
```

Sprawdź historię transakcji:
```bash
python3 check_pairs.py
```

## Bezpieczeństwo

- Bot używa bezpiecznego przechowywania kluczy API
- Implementuje limity ryzyka i ekspozycji
- Automatycznie zabezpiecza zyski
- Monitoruje i loguje wszystkie transakcje

## Licencja

MIT License 