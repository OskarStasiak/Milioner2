import json
import jwt
import time
import requests

# Wczytaj klucz z pliku JSON
with open("cdp_api_key.json") as f:
    key_data = json.load(f)

API_KEY = key_data["api_key_id"]
API_SECRET = key_data["api_key_secret"]

def generate_jwt():
    timestamp = int(time.time())
    expires = timestamp + 60
    header = {"alg": "ES256", "typ": "JWT"}
    payload = {
        "sub": API_KEY,
        "iat": timestamp,
        "exp": expires,
        "aud": ["retail_rest_api_pro"]
    }
    token = jwt.encode(payload, API_SECRET, algorithm="ES256", headers=header)
    print("\n=== JWT Token ===")
    print(f"Header: {json.dumps(header, indent=2)}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print(f"Token: {token}")
    return token

def get_balance():
    jwt_token = generate_jwt()
    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Content-Type": "application/json"
    }
    url = "https://api.coinbase.com/api/v3/brokerage/accounts"
    response = requests.get(url, headers=headers)
    print(f"\nStatus: {response.status_code}")
    print(f"Headers: {json.dumps(dict(response.headers), indent=2)}")
    print(f"Response: {response.text}")

if __name__ == "__main__":
    get_balance() 