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
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}")

def display_trade_details(trade):
    """Display detailed information about a trade in a readable format."""
    if not trade:
        return

    log_message("\n=== SZCZEGÓŁY TRANSAKCJI ===")
    
    # Basic trade information
    log_message(f"ID transakcji: {trade.get('id', 'N/A')}")
    log_message(f"Status: {trade.get('status', 'N/A')}")
    
    # Amounts
    amounts = trade.get('user_entered_amount', {})
    log_message(f"\nKwota wprowadzona przez użytkownika: {amounts.get('value', 'N/A')} {amounts.get('currency', 'N/A')}")
    
    amounts = trade.get('amount', {})
    log_message(f"Kwota transakcji: {amounts.get('value', 'N/A')} {amounts.get('currency', 'N/A')}")
    
    amounts = trade.get('subtotal', {})
    log_message(f"Kwota częściowa: {amounts.get('value', 'N/A')} {amounts.get('currency', 'N/A')}")
    
    amounts = trade.get('total', {})
    log_message(f"Kwota całkowita: {amounts.get('value', 'N/A')} {amounts.get('currency', 'N/A')}")
    
    # Fees
    log_message("\n=== OPŁATY ===")
    fees = trade.get('fees', [])
    for fee in fees:
        log_message(f"\nOpłata: {fee.get('title', 'N/A')}")
        log_message(f"Opis: {fee.get('description', 'N/A')}")
        amount = fee.get('amount', {})
        log_message(f"Kwota: {amount.get('value', 'N/A')} {amount.get('currency', 'N/A')}")
        if 'waived_details' in fee:
            waived = fee['waived_details'].get('amount', {})
            log_message(f"Opłata umorzona: {waived.get('value', 'N/A')} {waived.get('currency', 'N/A')}")
    
    # Exchange rate
    exchange_rate = trade.get('exchange_rate', {})
    log_message(f"\nKurs wymiany: {exchange_rate.get('value', 'N/A')} {exchange_rate.get('currency', 'N/A')}")
    
    # Subscription info
    sub_info = trade.get('subscription_info', {})
    if sub_info:
        log_message("\n=== INFORMACJE O SUB SKRYPCJI ===")
        log_message(f"Data resetu darmowego handlu: {sub_info.get('free_trading_reset_date', 'N/A')}")
        used = sub_info.get('used_zero_fee_trading', {})
        log_message(f"Wykorzystany darmowy handel: {used.get('value', 'N/A')} {used.get('currency', 'N/A')}")
        remaining = sub_info.get('remaining_free_trading_volume', {})
        log_message(f"Pozostały darmowy handel: {remaining.get('value', 'N/A')} {remaining.get('currency', 'N/A')}")
    
    # User warnings
    warnings = trade.get('user_warnings', [])
    if warnings:
        log_message("\n=== OSTRZEŻENIA ===")
        for warning in warnings:
            log_message(f"- {warning.get('message', 'N/A')}")

def get_trade(trade_id):
    try:
        load_dotenv('production.env')
        api_key_name = os.getenv('COINBASE_API_KEY_NAME')
        api_key_secret = os.getenv('COINBASE_API_KEY_SECRET')
        if not api_key_name or not api_key_secret:
            log_message("Brak wymaganych zmiennych środowiskowych")
            return None
        timestamp = str(int(time.time()))
        path = f'/api/v3/brokerage/convert/trade/{trade_id}'
        message = timestamp + 'GET' + path
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
        context = ssl._create_unverified_context()
        conn = http.client.HTTPSConnection("api.coinbase.com", context=context)
        conn.request("GET", path, '', headers)
        response = conn.getresponse()
        data = response.read()
        if response.status == 200:
            result = json.loads(data.decode("utf-8"))
            if 'trade' in result:
                display_trade_details(result['trade'])
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

def main():
    # UZUPEŁNIJ DANYMI SWOJEJ TRANSAKCJI:
    trade_id = "TWOJE_TRADE_ID"
    log_message(f"Pobieram informacje o transakcji {trade_id}...")
    get_trade(trade_id)

if __name__ == "__main__":
    main() 