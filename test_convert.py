import http.client
import json
import os
from dotenv import load_dotenv
import hmac
import hashlib
import time
from datetime import datetime
import ssl

def log_message(message):
    """Loguje wiadomość z timestampem."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}")

def convert_currency(from_currency, to_currency, amount):
    """Wykonuje konwersję walut."""
    try:
        # Ładowanie zmiennych środowiskowych
        load_dotenv('production.env')
        
        # Pobieranie kluczy API
        api_key_name = os.getenv('COINBASE_API_KEY_NAME')
        api_key_secret = os.getenv('COINBASE_API_KEY_SECRET')
        
        if not api_key_name or not api_key_secret:
            log_message("Brak wymaganych zmiennych środowiskowych")
            return None
            
        # Przygotowanie danych konwersji
        payload = {
            "from": from_currency,
            "to": to_currency,
            "amount": str(amount)
        }
        
        # Przygotowanie nagłówków
        timestamp = str(int(time.time()))
        path = '/api/v3/brokerage/convert'
        message = timestamp + 'POST' + path + json.dumps(payload)
        
        signature = hmac.new(
            api_key_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        headers = {
            'Content-Type': 'application/json',
            'CB-ACCESS-KEY': api_key_name.split('/')[-1],
            'CB-ACCESS-SIGN': signature,
            'CB-ACCESS-TIMESTAMP': timestamp,
            'CB-ORG-ID': api_key_name.split('/')[1]
        }
        
        # Tworzenie kontekstu SSL
        context = ssl._create_unverified_context()
        
        # Tworzenie połączenia
        conn = http.client.HTTPSConnection("api.coinbase.com", context=context)
        
        # Wykonanie zapytania
        conn.request("POST", path, json.dumps(payload), headers)
        
        # Pobranie odpowiedzi
        response = conn.getresponse()
        data = response.read()
        
        # Sprawdzenie statusu odpowiedzi
        if response.status == 200:
            result = json.loads(data.decode("utf-8"))
            log_message("Odpowiedź z API:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return result
        else:
            log_message(f"Błąd API: {response.status}")
            log_message(f"Treść odpowiedzi: {data.decode('utf-8')}")
            return None
            
    except Exception as e:
        log_message(f"Wystąpił błąd: {str(e)}")
        return None
    finally:
        try:
            conn.close()
        except:
            pass

def display_trade_details(trade):
    """Wyświetla szczegółowe informacje o transakcji."""
    log_message("\nSzczegóły transakcji:")
    log_message(f"ID transakcji: {trade.get('id', 'N/A')}")
    log_message(f"Status: {trade.get('status', 'N/A')}")
    
    # Kwoty
    user_entered = trade.get('user_entered_amount', {})
    log_message(f"\nKwota wprowadzona przez użytkownika: {user_entered.get('value', '0')} {user_entered.get('currency', 'N/A')}")
    
    amount = trade.get('amount', {})
    log_message(f"Kwota transakcji: {amount.get('value', '0')} {amount.get('currency', 'N/A')}")
    
    subtotal = trade.get('subtotal', {})
    log_message(f"Suma częściowa: {subtotal.get('value', '0')} {subtotal.get('currency', 'N/A')}")
    
    total = trade.get('total', {})
    log_message(f"Suma całkowita: {total.get('value', '0')} {total.get('currency', 'N/A')}")
    
    # Kurs wymiany
    exchange_rate = trade.get('exchange_rate', {})
    log_message(f"\nKurs wymiany: {exchange_rate.get('value', '0')} {exchange_rate.get('currency', 'N/A')}")
    
    # Opłaty
    fees = trade.get('fees', [])
    if fees:
        log_message("\nOpłaty:")
        for fee in fees:
            fee_amount = fee.get('amount', {})
            log_message(f"  - {fee.get('title', 'N/A')}: {fee_amount.get('value', '0')} {fee_amount.get('currency', 'N/A')}")
            if fee.get('description'):
                log_message(f"    Opis: {fee.get('description')}")
    
    # Całkowita opłata
    total_fee = trade.get('total_fee', {})
    if total_fee:
        fee_amount = total_fee.get('amount', {})
        log_message(f"\nCałkowita opłata: {fee_amount.get('value', '0')} {fee_amount.get('currency', 'N/A')}")
        if total_fee.get('description'):
            log_message(f"Opis: {total_fee.get('description')}")
    
    # Informacje o subskrypcji
    subscription = trade.get('subscription_info', {})
    if subscription:
        log_message("\nInformacje o subskrypcji:")
        if subscription.get('applied_subscription_benefit'):
            log_message("Zastosowano korzyści z subskrypcji")
            remaining = subscription.get('remaining_free_trading_volume', {})
            log_message(f"Pozostała darmowa kwota handlu: {remaining.get('value', '0')} {remaining.get('currency', 'N/A')}")
    
    # Ostrzeżenia
    warnings = trade.get('user_warnings', [])
    if warnings:
        log_message("\nOstrzeżenia:")
        for warning in warnings:
            log_message(f"  - {warning.get('message', 'N/A')}")
            if warning.get('link'):
                log_message(f"    Link: {warning.get('link', {}).get('url', 'N/A')}")

def main():
    """Główna funkcja programu."""
    # Przykładowa konwersja BTC na USD
    from_currency = "BTC"
    to_currency = "USD"
    amount = 0.001  # 0.001 BTC
    
    log_message(f"Próba konwersji {amount} {from_currency} na {to_currency}...")
    result = convert_currency(from_currency, to_currency, amount)
    
    if result and 'trade_id' in result:
        log_message(f"ID transakcji: {result['trade_id']}")
        log_message("Możesz teraz użyć tego ID w skrypcie test_trade.py")
    elif result and 'trade' in result:
        display_trade_details(result['trade'])
    else:
        log_message("Nie udało się pobrać szczegółów transakcji")

if __name__ == "__main__":
    main() 