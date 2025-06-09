import os
import json
import time
import logging
import hmac
import hashlib
import base64
from datetime import datetime
import requests
from urllib.parse import urlencode

class CoinbaseClient:
    def __init__(self, api_key, api_secret):
        """
        Inicjalizacja klienta REST API Coinbase.
        
        Args:
            api_key (str): Klucz API
            api_secret (str): Klucz prywatny
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.exchange.coinbase.com"
        self.logger = logging.getLogger('CoinbaseClient')
        
    def _get_signature(self, timestamp, method, request_path, body=''):
        """Generuje podpis dla żądania API."""
        message = f"{timestamp}{method}{request_path}{body}"
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
        
    def _request(self, method, endpoint, params=None, data=None):
        """Wykonuje żądanie do API."""
        try:
            url = f"{self.base_url}{endpoint}"
            timestamp = str(int(time.time()))
            
            # Przygotuj nagłówki
            headers = {
                'CB-ACCESS-KEY': self.api_key,
                'CB-ACCESS-SIGN': self._get_signature(timestamp, method, endpoint, json.dumps(data) if data else ''),
                'CB-ACCESS-TIMESTAMP': timestamp,
                'Content-Type': 'application/json'
            }
            
            # Wykonaj żądanie
            response = requests.request(
                method,
                url,
                headers=headers,
                params=params,
                json=data
            )
            
            # Sprawdź odpowiedź
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            self.logger.error(f"Błąd podczas wykonywania żądania API: {e}")
            raise
            
    def get_accounts(self):
        """Pobiera listę kont."""
        return self._request('GET', '/accounts')
        
    def get_account(self, account_id):
        """Pobiera szczegóły konta."""
        return self._request('GET', f'/accounts/{account_id}')
        
    def get_product_ticker(self, product_id):
        """Pobiera aktualny ticker dla produktu."""
        return self._request('GET', f'/products/{product_id}/ticker')
        
    def create_order(self, product_id, side, order_type, size=None, price=None, client_order_id=None):
        """
        Tworzy nowe zlecenie.
        
        Args:
            product_id (str): ID produktu (np. 'BTC-USD')
            side (str): 'buy' lub 'sell'
            order_type (str): 'market' lub 'limit'
            size (float): Rozmiar zlecenia
            price (float): Cena dla zleceń limit
            client_order_id (str): Opcjonalne ID zlecenia klienta
        """
        data = {
            'product_id': product_id,
            'side': side,
            'type': order_type
        }
        
        if size:
            data['size'] = str(size)
        if price:
            data['price'] = str(price)
        if client_order_id:
            data['client_order_id'] = client_order_id
            
        return self._request('POST', '/orders', data=data)
        
    def cancel_order(self, order_id):
        """Anuluje zlecenie."""
        return self._request('DELETE', f'/orders/{order_id}')
        
    def get_order(self, order_id):
        """Pobiera szczegóły zlecenia."""
        return self._request('GET', f'/orders/{order_id}')
        
    def get_orders(self, status=None, product_id=None):
        """Pobiera listę zleceń."""
        params = {}
        if status:
            params['status'] = status
        if product_id:
            params['product_id'] = product_id
        return self._request('GET', '/orders', params=params)
        
    def get_fills(self, order_id=None, product_id=None):
        """Pobiera historię wypełnień."""
        params = {}
        if order_id:
            params['order_id'] = order_id
        if product_id:
            params['product_id'] = product_id
        return self._request('GET', '/fills', params=params)
        
    def get_product_candles(self, product_id, start=None, end=None, granularity=60):
        """
        Pobiera świeczki dla produktu.
        
        Args:
            product_id (str): ID produktu
            start (str): Czas rozpoczęcia (ISO 8601)
            end (str): Czas zakończenia (ISO 8601)
            granularity (int): Interwał w sekundach (60, 300, 900, 3600, 21600, 86400)
        """
        params = {'granularity': granularity}
        if start:
            params['start'] = start
        if end:
            params['end'] = end
        return self._request('GET', f'/products/{product_id}/candles', params=params) 