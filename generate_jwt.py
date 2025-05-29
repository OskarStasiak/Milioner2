import jwt
from cryptography.hazmat.primitives import serialization
import time
import secrets

# Dane klucza API
key_name = "organizations/7ffbb587-c89b-49a7-8e49-45186964fa12/apiKeys/afd4070c-a484-4c53-98a4-080efcef64d9"
key_secret = """-----BEGIN EC PRIVATE KEY-----
MHcCAQEEIEZTiD2K/cS22+DVJGr2DgWOnI5A0J4xofmZT1WAnWZ6oAoGCCqGSM49
AwEHoUQDQgAEsYrDLX+ABLRU2kb1fK88YL5Orbq28EC1IlUsznql8H7BH2vY6naj
0t3BQAYqKM0D7cH0m/vD2MdAfUJXEE8Pnw==
-----END EC PRIVATE KEY-----"""

# Parametry żądania
request_method = "GET"
request_host = "api.coinbase.com"
request_path = "/v2/accounts"

def build_jwt(uri):
    """Generuje token JWT dla autoryzacji CDP API."""
    try:
        private_key_bytes = key_secret.encode('utf-8')
        private_key = serialization.load_pem_private_key(private_key_bytes, password=None)
        
        # Przygotuj payload JWT
        jwt_payload = {
            'sub': key_name,
            'iss': "cdp",
            'nbf': int(time.time()),
            'exp': int(time.time()) + 120,
            'uri': uri,
        }
        
        # Wygeneruj token JWT
        jwt_token = jwt.encode(
            jwt_payload,
            private_key,
            algorithm='ES256',
            headers={'kid': key_name, 'nonce': secrets.token_hex()},
        )
        
        return jwt_token
        
    except Exception as e:
        print(f"Błąd podczas generowania JWT: {str(e)}")
        return None

def main():
    # Przygotuj URI
    uri = f"{request_method} {request_host}{request_path}"
    
    # Wygeneruj token JWT
    jwt_token = build_jwt(uri)
    
    if jwt_token:
        print(f"export JWT={jwt_token}")
    else:
        print("Nie udało się wygenerować tokenu JWT")

if __name__ == "__main__":
    main() 