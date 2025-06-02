# Załaduj zmienne środowiskowe
from pathlib import Path
load_dotenv(str(Path(__file__).parent / 'production.env')) 

def print_detailed_balances():
    """Wyświetla szczegółowe informacje o saldach konta."""
    try:
        print("\n=== SPRAWDZANIE SALD KONTA ===")
        print("Pobieranie danych...")
        
        balances = get_all_balances()
        if not balances:
            print("Nie udało się pobrać sald - brak danych")
            return
            
        print("\n{:<10} {:<15} {:<15} {:<15}".format(
            "Waluta", "Dostępne", "Całkowite", "Zablokowane"
        ))
        print("-" * 60)
        
        total_value_usdc = 0
        
        for currency, data in balances.items():
            try:
                available = float(data['available'])
                total = float(data['total'])
                hold = float(data['hold'])
                
                # Jeśli to nie USDC, sprawdź czy para handlowa istnieje
                if currency != 'USDC':
                    pair = f"{currency}-USDC"
                    if pair in TRADING_PAIRS:
                        try:
                            current_price = get_product_ticker(pair)
                            if current_price is not None:
                                value_usdc = total * current_price
                                total_value_usdc += value_usdc
                            else:
                                value_usdc = 0
                        except:
                            value_usdc = 0
                    else:
                        value_usdc = 0
                else:
                    value_usdc = total
                    total_value_usdc += value_usdc
                
                print("{:<10} {:<15.8f} {:<15.8f} {:<15.8f}".format(
                    currency,
                    available,
                    total,
                    hold
                ))
                
                if currency != 'USDC' and value_usdc > 0:
                    print(f"Wartość w USDC: {value_usdc:.2f}")
                    
            except Exception as e:
                print(f"Błąd przy przetwarzaniu salda {currency}: {e}")
                continue
        
        print("\n=== PODSUMOWANIE ===")
        print(f"Całkowita wartość w USDC: {total_value_usdc:.2f}")
        print("=========================")
        
    except Exception as e:
        print(f"Błąd podczas pobierania sald: {str(e)}")
        print(f"Typ błędu: {type(e).__name__}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}") 

def get_all_balances():
    """Pobierz wszystkie salda konta."""
    try:
        accounts = client.get_accounts().accounts
        balances = {}
        for account in accounts:
            if hasattr(account, 'available_balance') and account.available_balance:
                currency = account.currency
                # Sprawdź czy waluta jest obsługiwana
                if currency not in ['USDC', 'USD', 'BTC', 'ETH']:
                    continue
                    
                # Obsługa różnych formatów danych
                if isinstance(account.available_balance, dict):
                    available = float(account.available_balance.get('value', 0))
                else:
                    available = float(account.available_balance.value)
                
                if hasattr(account, 'total_balance'):
                    if isinstance(account.total_balance, dict):
                        total = float(account.total_balance.get('value', 0))
                    else:
                        total = float(account.total_balance.value)
                else:
                    total = available
                
                if hasattr(account, 'hold'):
                    if isinstance(account.hold, dict):
                        hold = float(account.hold.get('value', 0))
                    else:
                        hold = float(account.hold.value)
                else:
                    hold = 0

                balances[currency] = {
                    'available': available,
                    'total': total,
                    'hold': hold,
                    'currency': currency,
                    'type': getattr(account, 'type', 'unknown'),
                    'active': getattr(account, 'active', True)
                }
        return balances
    except Exception as e:
        logger.error(f"Błąd podczas pobierania sald: {e}")
        return {} 

def check_balances():
    """Sprawdza salda konta używając bezpośrednio API Coinbase."""
    try:
        print("\n=== SPRAWDZANIE SALD KONTA ===")
        accounts = client.get_accounts().accounts
        
        print("\n{:<10} {:<15} {:<15}".format(
            "Waluta", "Dostępne", "Całkowite"
        ))
        print("-" * 45)
        
        for account in accounts:
            if hasattr(account, 'currency') and hasattr(account, 'available_balance'):
                currency = account.currency
                if currency in ['USDC', 'USD', 'BTC', 'ETH']:
                    available = float(account.available_balance.value)
                    total = float(account.total_balance.value) if hasattr(account, 'total_balance') else available
                    print("{:<10} {:<15.8f} {:<15.8f}".format(
                        currency,
                        available,
                        total
                    ))
        
        print("\n=========================")
    except Exception as e:
        print(f"Błąd podczas sprawdzania sald: {e}") 