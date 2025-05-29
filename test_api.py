import os
import http.client
import json
import ssl
import hmac
import hashlib
import time
from datetime import datetime
import jwt
import secrets
from cryptography.hazmat.primitives import serialization
import requests
from dotenv import load_dotenv
import base64

# Ładowanie zmiennych środowiskowych
load_dotenv('production.env')

# Pobieranie kluczy API
api_key = os.getenv('COINBASE_API_KEY')
api_secret = os.getenv('COINBASE_API_SECRET')

def log_message(message):
    """Loguje wiadomość z timestampem."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}")

def generate_jwt(method, path):
    """Generuje token JWT dla autoryzacji API."""
    try:
        load_dotenv('production.env')
        api_key_name = os.getenv('COINBASE_API_KEY_NAME')
        api_key_secret = os.getenv('COINBASE_API_KEY_SECRET')
        
        if not api_key_name or not api_key_secret:
            log_message("Brak wymaganych zmiennych środowiskowych")
            return None

        # Zamień \n na prawdziwe nowe linie
        api_key_secret = api_key_secret.replace('\\n', '\n')

        # Przygotuj klucz prywatny
        private_key = serialization.load_pem_private_key(
            api_key_secret.encode('utf-8'),
            password=None
        )
        
        # Przygotuj payload
        current_time = int(time.time())
        payload = {
            'sub': api_key_name,
            'iss': "cdp",
            'nbf': current_time,
            'exp': current_time + 120,  # Token ważny przez 2 minuty
            'uri': f"{method} {path}"
        }
        
        # Przygotuj nagłówki
        headers = {
            'kid': api_key_name,
            'nonce': hashlib.sha256(str(current_time).encode()).hexdigest()[:16]
        }
        
        # Wygeneruj token używając Ed25519
        token = jwt.encode(
            payload,
            private_key,
            algorithm='EdDSA',  # Zmiana algorytmu na EdDSA (Ed25519)
            headers=headers
        )
        
        return token
    except Exception as e:
        log_message(f"Błąd podczas generowania JWT: {str(e)}")
        return None

def make_api_request(method, endpoint, headers=None, data=None):
    """Wykonuje żądanie do API Coinbase z obsługą błędów."""
    try:
        # Przygotuj kontekst SSL
        context = ssl._create_unverified_context()
        
        # Utwórz połączenie
        conn = http.client.HTTPSConnection("api.coinbase.com", context=context)
        
        # Dodaj wymagane nagłówki
        if headers is None:
            headers = {}
            
        # Pobierz timestamp
        timestamp = str(int(time.time()))
        
        # Pobierz klucze API
        api_key = os.getenv('COINBASE_API_KEY_NAME')
        api_secret = os.getenv('COINBASE_API_KEY_SECRET')
        
        # Przygotuj podpis
        message = timestamp + method + endpoint + (data if data else '')
        signature = hmac.new(
            api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'CB-ACCESS-KEY': api_key,
            'CB-ACCESS-SIGN': signature,
            'CB-ACCESS-TIMESTAMP': timestamp,
            'CB-VERSION': '2023-12-01'
        })
        
        # Wykonaj żądanie
        conn.request(method, endpoint, data, headers)
        
        # Pobierz odpowiedź
        response = conn.getresponse()
        data = response.read().decode("utf-8")
        
        # Sprawdź status odpowiedzi
        if response.status != 200:
            log_message(f"Błąd API: {response.status} {response.reason}")
            log_message(f"Treść odpowiedzi: {data}")
            return None
            
        # Parsuj odpowiedź
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            log_message(f"Nieprawidłowa odpowiedź JSON: {data}")
            return None
            
    except Exception as e:
        log_message(f"Błąd podczas wykonywania żądania: {str(e)}")
        return None
    finally:
        if 'conn' in locals():
            conn.close()

def get_accounts(cursor=None):
    """Pobiera listę kont z obsługą paginacji."""
    try:
        # Przygotuj ścieżkę dla JWT
        path = "/api/v3/brokerage/accounts"
        if cursor:
            path += f"?cursor={cursor}"
        
        # Wygeneruj token JWT
        jwt_token = generate_jwt("GET", path)
        if not jwt_token:
            return None
        
        # Przygotuj nagłówki
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f"Bearer {jwt_token}"
        }
        
        # Przygotuj endpoint
        endpoint = path
            
        # Wykonaj żądanie
        return make_api_request("GET", endpoint, headers)
            
    except Exception as e:
        log_message(f"Błąd podczas pobierania kont: {str(e)}")
        return None

def test_direct_api():
    """Testuje bezpośrednie połączenie z API Coinbase."""
    try:
        all_accounts = []
        cursor = None
        
        while True:
            # Pobierz stronę kont
            response = get_accounts(cursor)
            if not response:
                break
                
            # Dodaj konta do listy
            if 'accounts' in response:
                accounts = response['accounts']
                if isinstance(accounts, list):
                    all_accounts.extend(accounts)
                else:
                    all_accounts.append(accounts)
                    
            # Sprawdź czy jest następna strona
            if not response.get('has_next'):
                break
                
            # Pobierz cursor do następnej strony
            cursor = response.get('cursor')
            if not cursor:
                break
                
        # Wyświetl wszystkie konta
        log_message(f"Znaleziono {len(all_accounts)} kont:")
        for account in all_accounts:
            currency = account.get('currency', 'UNKNOWN')
            name = account.get('name', 'UNKNOWN')
            available = account.get('available_balance', {}).get('value', '0')
            hold = account.get('hold', {}).get('value', '0')
            active = account.get('active', False)
            ready = account.get('ready', False)
            
            log_message(f"  - {currency} ({name}):")
            log_message(f"    Dostępne: {available}")
            log_message(f"    Zablokowane: {hold}")
            log_message(f"    Status: {'Aktywne' if active else 'Nieaktywne'}")
            log_message(f"    Gotowe: {'Tak' if ready else 'Nie'}")
            
    except Exception as e:
        log_message(f"Błąd podczas wykonywania żądania: {str(e)}")

def test_convert_currency():
    """Testuje konwersję walut dla wszystkich dostępnych par."""
    try:
        # Najpierw pobierz listę kont
        accounts_response = get_accounts()
        if not accounts_response or 'accounts' not in accounts_response:
            log_message("Nie udało się pobrać listy kont")
            return
            
        accounts = accounts_response['accounts']
        if not isinstance(accounts, list):
            accounts = [accounts]
            
        # Znajdź konta z dostępnymi środkami
        available_accounts = []
        for account in accounts:
            available = account.get('available_balance', {}).get('value', '0')
            if float(available) > 0:
                available_accounts.append({
                    'id': account.get('uuid'),
                    'currency': account.get('currency'),
                    'amount': available
                })
                
        if not available_accounts:
            log_message("Brak kont z dostępnymi środkami")
            return
            
        log_message(f"Znaleziono {len(available_accounts)} kont z dostępnymi środkami")
        
        # Testuj konwersję dla każdej pary walut
        for source in available_accounts:
            for target in accounts:
                # Pomijamy konwersję na tę samą walutę
                if source['currency'] == target.get('currency'):
                    continue
                    
                log_message(f"\nTest konwersji {source['currency']} -> {target.get('currency')}")
                
                # Przygotuj dane konwersji
                trade_data = {
                    "from_account_id": source['id'],
                    "to_account_id": target.get('uuid'),
                    "from_amount": source['amount'],
                    "to_currency": target.get('currency')
                }
                
                # Przygotuj ścieżkę dla JWT
                path = "/api/v3/brokerage/convert/trade"
                
                # Wygeneruj token JWT
                jwt_token = generate_jwt("POST", path)
                if not jwt_token:
                    log_message("Nie udało się wygenerować tokenu JWT")
                    continue
                
                # Przygotuj nagłówki
                headers = {
                    'Content-Type': 'application/json',
                    'Authorization': f"Bearer {jwt_token}"
                }
                
                try:
                    # Wykonaj żądanie
                    context = ssl._create_unverified_context()
                    conn = http.client.HTTPSConnection("api.coinbase.com", context=context)
                    conn.request("POST", path, json.dumps(trade_data), headers)
                    
                    # Pobierz odpowiedź
                    response = conn.getresponse()
                    data = response.read().decode("utf-8")
                    
                    # Parsuj odpowiedź
                    try:
                        result = json.loads(data)
                        trade = result.get('trade', {})
                        
                        # Podstawowe informacje
                        log_message(f"ID transakcji: {trade.get('id', 'N/A')}")
                        log_message(f"Status: {trade.get('status', 'N/A')}")
                        log_message(f"Waluta źródłowa: {trade.get('source_currency', 'N/A')}")
                        log_message(f"Waluta docelowa: {trade.get('target_currency', 'N/A')}")
                        
                        # Kwoty
                        user_amount = trade.get('user_entered_amount', {})
                        log_message(f"Kwota wprowadzona: {user_amount.get('value', '0')} {user_amount.get('currency', 'N/A')}")
                        
                        amount = trade.get('amount', {})
                        log_message(f"Kwota faktyczna: {amount.get('value', '0')} {amount.get('currency', 'N/A')}")
                        
                        subtotal = trade.get('subtotal', {})
                        log_message(f"Podsumowanie: {subtotal.get('value', '0')} {subtotal.get('currency', 'N/A')}")
                        
                        total = trade.get('total', {})
                        log_message(f"Całkowita kwota: {total.get('value', '0')} {total.get('currency', 'N/A')}")
                        
                        # Kurs wymiany
                        exchange_rate = trade.get('exchange_rate', {})
                        log_message(f"Kurs wymiany: {exchange_rate.get('value', '0')} {exchange_rate.get('currency', 'N/A')}")
                        
                        # Ceny jednostkowe
                        unit_price = trade.get('unit_price', {})
                        if unit_price:
                            target_to_fiat = unit_price.get('target_to_fiat', {})
                            if target_to_fiat:
                                amount = target_to_fiat.get('amount', {})
                                log_message(f"Cena docelowa w fiat: {amount.get('value', '0')} {amount.get('currency', 'N/A')}")
                            
                            target_to_source = unit_price.get('target_to_source', {})
                            if target_to_source:
                                amount = target_to_source.get('amount', {})
                                log_message(f"Cena docelowa w źródłowej: {amount.get('value', '0')} {amount.get('currency', 'N/A')}")
                            
                            source_to_fiat = unit_price.get('source_to_fiat', {})
                            if source_to_fiat:
                                amount = source_to_fiat.get('amount', {})
                                log_message(f"Cena źródłowa w fiat: {amount.get('value', '0')} {amount.get('currency', 'N/A')}")
                        
                        # Opłaty
                        fees = trade.get('fees', [])
                        if fees:
                            log_message("Opłaty:")
                            for fee in fees:
                                fee_amount = fee.get('amount', {})
                                log_message(f"  - {fee.get('title', 'N/A')}: {fee_amount.get('value', '0')} {fee_amount.get('currency', 'N/A')}")
                                if fee.get('description'):
                                    log_message(f"    Opis: {fee.get('description')}")
                                if fee.get('label'):
                                    log_message(f"    Etykieta: {fee.get('label')}")
                        
                        # Całkowita opłata
                        total_fee = trade.get('total_fee', {})
                        if total_fee:
                            fee_amount = total_fee.get('amount', {})
                            log_message(f"Całkowita opłata: {fee_amount.get('value', '0')} {fee_amount.get('currency', 'N/A')}")
                            if total_fee.get('description'):
                                log_message(f"Opis opłaty: {total_fee.get('description')}")
                        
                        # Opłata bez podatku
                        total_fee_without_tax = trade.get('total_fee_without_tax', {})
                        if total_fee_without_tax:
                            fee_amount = total_fee_without_tax.get('amount', {})
                            log_message(f"Opłata bez podatku: {fee_amount.get('value', '0')} {fee_amount.get('currency', 'N/A')}")
                        
                        # Szczegóły podatku
                        tax_details = trade.get('tax_details', [])
                        if tax_details:
                            log_message("Szczegóły podatku:")
                            for tax in tax_details:
                                tax_amount = tax.get('amount', {})
                                log_message(f"  - {tax.get('name', 'N/A')}: {tax_amount.get('value', '0')} {tax_amount.get('currency', 'N/A')}")
                        
                        # Informacje o subskrypcji
                        subscription_info = trade.get('subscription_info', {})
                        if subscription_info:
                            log_message("Informacje o subskrypcji:")
                            if subscription_info.get('has_benefit_cap') is not None:
                                log_message(f"  Limit korzyści: {'Tak' if subscription_info.get('has_benefit_cap') else 'Nie'}")
                            if subscription_info.get('applied_subscription_benefit') is not None:
                                log_message(f"  Zastosowano korzyść: {'Tak' if subscription_info.get('applied_subscription_benefit') else 'Nie'}")
                        
                        # Ostrzeżenia
                        warnings = trade.get('user_warnings', [])
                        if warnings:
                            log_message("Ostrzeżenia:")
                            for warning in warnings:
                                log_message(f"  - {warning.get('message', 'N/A')}")
                                if warning.get('code'):
                                    log_message(f"    Kod: {warning.get('code')}")
                        
                        # Powód anulowania
                        cancellation = trade.get('cancellation_reason', {})
                        if cancellation:
                            log_message("Powód anulowania:")
                            log_message(f"  Tytuł: {cancellation.get('title', 'N/A')}")
                            log_message(f"  Wiadomość: {cancellation.get('message', 'N/A')}")
                            log_message(f"  Kod: {cancellation.get('code', 'N/A')}")
                            
                    except json.JSONDecodeError:
                        log_message(f"Nieprawidłowa odpowiedź JSON: {data}")
                        
                except Exception as e:
                    log_message(f"Błąd podczas konwersji: {str(e)}")
                finally:
                    if 'conn' in locals():
                        conn.close()
                        
    except Exception as e:
        log_message(f"Błąd podczas testowania konwersji: {str(e)}")

def test_convert_quote():
    """Testuje wycenę konwersji walut."""
    try:
        # Najpierw pobierz listę kont
        accounts_response = get_accounts()
        if not accounts_response or 'accounts' not in accounts_response:
            log_message("Nie udało się pobrać listy kont")
            return
            
        accounts = accounts_response['accounts']
        if not isinstance(accounts, list):
            accounts = [accounts]
            
        # Znajdź konta z dostępnymi środkami
        available_accounts = []
        for account in accounts:
            available = account.get('available_balance', {}).get('value', '0')
            if float(available) > 0:
                available_accounts.append({
                    'id': account.get('uuid'),
                    'currency': account.get('currency'),
                    'amount': available
                })
                
        if not available_accounts:
            log_message("Brak kont z dostępnymi środkami")
            return
            
        log_message(f"Znaleziono {len(available_accounts)} kont z dostępnymi środkami")
        
        # Testuj wycenę dla każdej pary walut
        for source in available_accounts:
            for target in accounts:
                # Pomijamy konwersję na tę samą walutę
                if source['currency'] == target.get('currency'):
                    continue
                    
                log_message(f"\nTest wyceny konwersji {source['currency']} -> {target.get('currency')}")
                
                # Przygotuj dane wyceny
                quote_data = {
                    "from_account_id": source['id'],
                    "to_account_id": target.get('uuid'),
                    "from_amount": source['amount'],
                    "to_currency": target.get('currency')
                }
                
                # Przygotuj ścieżkę dla JWT
                path = "/api/v3/brokerage/convert/quote"
                
                # Wygeneruj token JWT
                jwt_token = generate_jwt("POST", path)
                if not jwt_token:
                    log_message("Nie udało się wygenerować tokenu JWT")
                    continue
                
                # Przygotuj nagłówki
                headers = {
                    'Content-Type': 'application/json',
                    'Authorization': f"Bearer {jwt_token}"
                }
                
                try:
                    # Wykonaj żądanie
                    context = ssl._create_unverified_context()
                    conn = http.client.HTTPSConnection("api.coinbase.com", context=context)
                    conn.request("POST", path, json.dumps(quote_data), headers)
                    
                    # Pobierz odpowiedź
                    response = conn.getresponse()
                    data = response.read().decode("utf-8")
                    
                    # Parsuj odpowiedź
                    try:
                        result = json.loads(data)
                        quote = result.get('quote', {})
                        
                        # Wyświetl szczegóły wyceny
                        log_message(f"ID wyceny: {quote.get('id', 'N/A')}")
                        log_message(f"Status: {quote.get('status', 'N/A')}")
                        
                        # Kwota źródłowa
                        from_amount = quote.get('from_amount', {})
                        log_message(f"Kwota źródłowa: {from_amount.get('value', '0')} {from_amount.get('currency', 'N/A')}")
                        
                        # Kwota docelowa
                        to_amount = quote.get('to_amount', {})
                        log_message(f"Kwota docelowa: {to_amount.get('value', '0')} {to_amount.get('currency', 'N/A')}")
                        
                        # Kurs wymiany
                        exchange_rate = quote.get('exchange_rate', {})
                        log_message(f"Kurs wymiany: {exchange_rate.get('value', '0')} {exchange_rate.get('currency', 'N/A')}")
                        
                    except json.JSONDecodeError:
                        log_message(f"Nieprawidłowa odpowiedź JSON: {data}")
                        
                except Exception as e:
                    log_message(f"Błąd podczas pobierania wyceny: {str(e)}")
                finally:
                    if 'conn' in locals():
                        conn.close()
                        
    except Exception as e:
        log_message(f"Błąd podczas testowania wyceny: {str(e)}")

def test_get_trade_details(trade_id):
    """Pobiera szczegóły transakcji konwersji."""
    try:
        # Przygotuj ścieżkę dla JWT
        path = f"/api/v3/brokerage/convert/trade/{trade_id}"
        
        # Wygeneruj token JWT
        jwt_token = generate_jwt("GET", path)
        if not jwt_token:
            log_message("Nie udało się wygenerować tokenu JWT")
            return
        
        # Przygotuj nagłówki
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f"Bearer {jwt_token}"
        }
        
        try:
            # Wykonaj żądanie
            context = ssl._create_unverified_context()
            conn = http.client.HTTPSConnection("api.coinbase.com", context=context)
            conn.request("GET", path, "", headers)
            
            # Pobierz odpowiedź
            response = conn.getresponse()
            data = response.read().decode("utf-8")
            
            # Parsuj odpowiedź
            try:
                result = json.loads(data)
                trade = result.get('trade', {})
                
                # Podstawowe informacje
                log_message(f"ID transakcji: {trade.get('id', 'N/A')}")
                log_message(f"Status: {trade.get('status', 'N/A')}")
                log_message(f"Waluta źródłowa: {trade.get('source_currency', 'N/A')}")
                log_message(f"Waluta docelowa: {trade.get('target_currency', 'N/A')}")
                
                # Kwoty
                user_amount = trade.get('user_entered_amount', {})
                log_message(f"Kwota wprowadzona: {user_amount.get('value', '0')} {user_amount.get('currency', 'N/A')}")
                
                amount = trade.get('amount', {})
                log_message(f"Kwota faktyczna: {amount.get('value', '0')} {amount.get('currency', 'N/A')}")
                
                subtotal = trade.get('subtotal', {})
                log_message(f"Podsumowanie: {subtotal.get('value', '0')} {subtotal.get('currency', 'N/A')}")
                
                total = trade.get('total', {})
                log_message(f"Całkowita kwota: {total.get('value', '0')} {total.get('currency', 'N/A')}")
                
                # Kurs wymiany
                exchange_rate = trade.get('exchange_rate', {})
                log_message(f"Kurs wymiany: {exchange_rate.get('value', '0')} {exchange_rate.get('currency', 'N/A')}")
                
                # Ceny jednostkowe
                unit_price = trade.get('unit_price', {})
                if unit_price:
                    target_to_fiat = unit_price.get('target_to_fiat', {})
                    if target_to_fiat:
                        amount = target_to_fiat.get('amount', {})
                        log_message(f"Cena docelowa w fiat: {amount.get('value', '0')} {amount.get('currency', 'N/A')}")
                    
                    target_to_source = unit_price.get('target_to_source', {})
                    if target_to_source:
                        amount = target_to_source.get('amount', {})
                        log_message(f"Cena docelowa w źródłowej: {amount.get('value', '0')} {amount.get('currency', 'N/A')}")
                    
                    source_to_fiat = unit_price.get('source_to_fiat', {})
                    if source_to_fiat:
                        amount = source_to_fiat.get('amount', {})
                        log_message(f"Cena źródłowa w fiat: {amount.get('value', '0')} {amount.get('currency', 'N/A')}")
                
                # Opłaty
                fees = trade.get('fees', [])
                if fees:
                    log_message("Opłaty:")
                    for fee in fees:
                        fee_amount = fee.get('amount', {})
                        log_message(f"  - {fee.get('title', 'N/A')}: {fee_amount.get('value', '0')} {fee_amount.get('currency', 'N/A')}")
                        if fee.get('description'):
                            log_message(f"    Opis: {fee.get('description')}")
                        if fee.get('label'):
                            log_message(f"    Etykieta: {fee.get('label')}")
                
                # Całkowita opłata
                total_fee = trade.get('total_fee', {})
                if total_fee:
                    fee_amount = total_fee.get('amount', {})
                    log_message(f"Całkowita opłata: {fee_amount.get('value', '0')} {fee_amount.get('currency', 'N/A')}")
                    if total_fee.get('description'):
                        log_message(f"Opis opłaty: {total_fee.get('description')}")
                
                # Opłata bez podatku
                total_fee_without_tax = trade.get('total_fee_without_tax', {})
                if total_fee_without_tax:
                    fee_amount = total_fee_without_tax.get('amount', {})
                    log_message(f"Opłata bez podatku: {fee_amount.get('value', '0')} {fee_amount.get('currency', 'N/A')}")
                
                # Szczegóły podatku
                tax_details = trade.get('tax_details', [])
                if tax_details:
                    log_message("Szczegóły podatku:")
                    for tax in tax_details:
                        tax_amount = tax.get('amount', {})
                        log_message(f"  - {tax.get('name', 'N/A')}: {tax_amount.get('value', '0')} {tax_amount.get('currency', 'N/A')}")
                
                # Informacje o subskrypcji
                subscription_info = trade.get('subscription_info', {})
                if subscription_info:
                    log_message("Informacje o subskrypcji:")
                    if subscription_info.get('has_benefit_cap') is not None:
                        log_message(f"  Limit korzyści: {'Tak' if subscription_info.get('has_benefit_cap') else 'Nie'}")
                    if subscription_info.get('applied_subscription_benefit') is not None:
                        log_message(f"  Zastosowano korzyść: {'Tak' if subscription_info.get('applied_subscription_benefit') else 'Nie'}")
                
                # Ostrzeżenia
                warnings = trade.get('user_warnings', [])
                if warnings:
                    log_message("Ostrzeżenia:")
                    for warning in warnings:
                        log_message(f"  - {warning.get('message', 'N/A')}")
                        if warning.get('code'):
                            log_message(f"    Kod: {warning.get('code')}")
                
                # Powód anulowania
                cancellation = trade.get('cancellation_reason', {})
                if cancellation:
                    log_message("Powód anulowania:")
                    log_message(f"  Tytuł: {cancellation.get('title', 'N/A')}")
                    log_message(f"  Wiadomość: {cancellation.get('message', 'N/A')}")
                    log_message(f"  Kod: {cancellation.get('code', 'N/A')}")
                    
            except json.JSONDecodeError:
                log_message(f"Nieprawidłowa odpowiedź JSON: {data}")
                
        except Exception as e:
            log_message(f"Błąd podczas pobierania szczegółów transakcji: {str(e)}")
        finally:
            if 'conn' in locals():
                conn.close()
                
    except Exception as e:
        log_message(f"Błąd podczas testowania pobierania szczegółów transakcji: {str(e)}")

def test_key_permissions():
    """Testuje uprawnienia klucza API."""
    log_message("\nTest uprawnień klucza API:")
    
    # Pobierz informacje o kluczu
    response = make_api_request("GET", "/api/v3/brokerage/accounts")
    
    if response and "accounts" in response:
        for account in response["accounts"]:
            log_message(f"\nKonto: {account.get('name', 'N/A')}")
            log_message(f"ID: {account.get('uuid', 'N/A')}")
            log_message(f"Typ: {account.get('type', 'N/A')}")
            log_message(f"Status: {account.get('active', 'N/A')}")
            log_message(f"Może handlować: {account.get('can_trade', 'N/A')}")
            log_message(f"Może przeglądać: {account.get('can_view', 'N/A')}")
            log_message(f"Może transferować: {account.get('can_transfer', 'N/A')}")
            log_message(f"Portfolio UUID: {account.get('portfolio_uuid', 'N/A')}")
            log_message(f"Typ portfolio: {account.get('portfolio_type', 'N/A')}")
    else:
        log_message("Nie udało się pobrać informacji o uprawnieniach klucza API")

def test_endpoint(endpoint, method='GET', body=''):
    """Testuje endpoint API z autoryzacją JWT."""
    try:
        # Wygeneruj token JWT
        jwt_token = generate_jwt(method, endpoint)
        if not jwt_token:
            return False
            
        # Przygotuj nagłówki
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f"Bearer {jwt_token}"
        }
        
        # Wyświetl informacje debugowe
        log_message(f"\n=== TEST ENDPOINTU: {endpoint} ===")
        log_message(f"Method: {method}")
        log_message(f"Headers: {json.dumps(headers, indent=2)}")
        
        # Wykonaj żądanie
        context = ssl._create_unverified_context()
        conn = http.client.HTTPSConnection("api.coinbase.com", context=context)
        conn.request(method, endpoint, body, headers)
        response = conn.getresponse()
        data = response.read()
        
        log_message(f"Status: {response.status}")
        log_message(f"Odpowiedź: {data.decode('utf-8')}")
        
        return response.status == 200
    except Exception as e:
        log_message(f"Wystąpił błąd: {str(e)}")
        return False
    finally:
        try:
            conn.close()
        except:
            pass

def main():
    try:
        # Test bezpośredniego połączenia z API
        test_direct_api()
            
        # Test konwersji walut
        test_convert_currency()
            
        # Test wyceny konwersji walut
        test_convert_quote()
        
        # Test pobierania szczegółów transakcji (przykładowe ID)
        test_get_trade_details("example_trade_id")
        
        # Test sprawdzania uprawnień klucza API
        test_key_permissions()
            
    except Exception as e:
        log_message(f"Błąd podczas wykonywania żądania: {str(e)}")

if __name__ == "__main__":
    print("Testowanie połączenia z API Coinbase...")
    get_accounts()

# For instructions generating JWT, check the "API Key Authentication" section
JWT_TOKEN = "eyJhbGciOiJFUzI1NiIsImtpZCI6ImFmZDQwNzBjLWE0ODQtNGM1My05OGE0LTA4MGVmY2VmNjRkOSIsIm5vbmNlIjoiODhlYmM3Zjc3YjMyMjc3YjMyYjU3MDA4M2M1MWI0NDkiLCJ0eXAiOiJKV1QifQ.eyJzdWIiOiJvcmdhbml6YXRpb25zLzdmZmJiNTg3LWM4OWItNDlhNy04ZTQ5LTQ1MTg2OTY0ZmExMi9hcGlLZXlzL2FmZDQwNzBjLWE0ODQtNGM1My05OGE0LTA4MGVmY2VmNjRkOSIsImlzcyI6ImNkcCIsIm5iZiI6MTc0Nzg0OTU4MiwiZXhwIjoxNzQ3ODQ5NzAyLCJ1cmkiOiJHRVQgL3YzL2Jyb2tlcmFnZS9hY2NvdW50cyJ9.FWGxeKmAwyU801l0h0pDIa8qUQhpUuVvW0H8TR3wM2RB669B0QTyQdgC7T2WqB8ZC4296yBwFp_5FNEaa20z8A"

ENDPOINT_URL = "https://api.coinbase.com/v2/accounts/f603f97c-37d7-4e58-b264-c27e9e393dd9/addresses"

def get_addresses():
    # Generate headers with JWT for authentication
    headers = {
        "Authorization": f"Bearer {JWT_TOKEN}",
        "CB-ACCESS-KEY": "43f43b13-4cd0-41a3-9db3-37a90fcafd1d",
        "CB-ORG-ID": "7ffbb587-c89b-49a7-8e49-45186964fa12"
    }

    # Make the API request
    response = requests.get(ENDPOINT_URL, headers=headers)

    return response.json()  # Return the JSON response

addresses = get_addresses()
print(addresses) 