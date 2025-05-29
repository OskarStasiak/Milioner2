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

def display_summary(summary):
    """Wyświetla podsumowanie transakcji w czytelnym formacie."""
    if not summary:
        return

    log_message("\n=== PODSUMOWANIE TRANSAKCJI ===")
    
    # Całkowite wolumeny i opłaty
    log_message("\nCałkowite wolumeny i opłaty:")
    log_message(f"Całkowity wolumen: {summary.get('total_volume', 'N/A')} USD")
    log_message(f"Całkowite opłaty: {summary.get('total_fees', 'N/A')} USD")
    log_message(f"Całkowite saldo: {summary.get('total_balance', 'N/A')} USD")
    
    # Poziomy opłat
    fee_tier = summary.get('fee_tier', {})
    if fee_tier:
        log_message("\nPoziomy opłat:")
        log_message(f"Poziom cenowy: {fee_tier.get('pricing_tier', 'N/A')}")
        log_message(f"Zakres USD: {fee_tier.get('usd_from', 'N/A')} - {fee_tier.get('usd_to', 'N/A')}")
        log_message(f"Opłata taker: {float(fee_tier.get('taker_fee_rate', 0)) * 100:.2f}%")
        log_message(f"Opłata maker: {float(fee_tier.get('maker_fee_rate', 0)) * 100:.2f}%")
        log_message(f"Zakres AOP: {fee_tier.get('aop_from', 'N/A')} - {fee_tier.get('aop_to', 'N/A')}")
        log_message(f"Zakres wolumenu perps: {fee_tier.get('perps_vol_from', 'N/A')} - {fee_tier.get('perps_vol_to', 'N/A')}")
    
    # Wolumeny i opłaty dla różnych platform
    log_message("\nWolumeny i opłaty dla platform:")
    log_message(f"Advanced Trade - wolumen: {summary.get('advanced_trade_only_volume', 'N/A')} USD")
    log_message(f"Advanced Trade - opłaty: {summary.get('advanced_trade_only_fees', 'N/A')} USD")
    log_message(f"Coinbase Pro - wolumen: {summary.get('coinbase_pro_volume', 'N/A')} USD")
    log_message(f"Coinbase Pro - opłaty: {summary.get('coinbase_pro_fees', 'N/A')} USD")
    
    # Podatek od towarów i usług
    gst = summary.get('goods_and_services_tax', {})
    if gst:
        log_message("\nPodatek GST:")
        log_message(f"Stawka: {gst.get('rate', 'N/A')}")
        log_message(f"Typ: {gst.get('type', 'N/A')}")
    
    # Stopa marży
    if 'margin_rate' in summary:
        log_message(f"\nStopa marży: {summary.get('margin_rate', 0) * 100:.2f}%")

def get_transaction_summary():
    try:
        load_dotenv('production.env')
        api_key_name = os.getenv('COINBASE_API_KEY_NAME')
        api_key_secret = os.getenv('COINBASE_API_KEY_SECRET')
        
        if not api_key_name or not api_key_secret:
            log_message("Brak wymaganych zmiennych środowiskowych")
            return None

        timestamp = str(int(time.time()))
        method = 'GET'
        path = '/api/v3/brokerage/transaction_summary'
        body = ''
        
        # Przygotowanie wiadomości do podpisu
        message = f"{timestamp}{method}{path}{body}"
        
        # Generowanie podpisu
        signature = hmac.new(
            api_key_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        # Przygotowanie nagłówków
        headers = {
            'Content-Type': 'application/json',
            'CB-ACCESS-KEY': api_key_name.split('/')[-1],
            'CB-ACCESS-SIGN': signature,
            'CB-ACCESS-TIMESTAMP': timestamp,
            'CB-ORG-ID': api_key_name.split('/')[1]
        }

        # Wyświetlenie informacji debugowych
        log_message("\n=== INFORMACJE DEBUGOWE ===")
        log_message(f"Timestamp: {timestamp}")
        log_message(f"Message: {message}")
        log_message(f"Signature: {signature}")
        log_message(f"Headers: {json.dumps(headers, indent=2)}")

        context = ssl._create_unverified_context()
        conn = http.client.HTTPSConnection("api.coinbase.com", context=context)
        conn.request(method, path, body, headers)
        response = conn.getresponse()
        data = response.read()

        if response.status == 200:
            result = json.loads(data.decode("utf-8"))
            display_summary(result)
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
    log_message("Pobieranie podsumowania transakcji...")
    get_transaction_summary()

if __name__ == "__main__":
    main() 