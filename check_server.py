import requests
import sys

try:
    print("Checking if server is running...")
    response = requests.get("http://localhost:8000/health")
    print(f"Status code: {response.status_code}")
    if response.status_code == 200:
        print("Server is running!")
        print(response.json())
    else:
        print(f"Server returned error: {response.text}")
except Exception as e:
    print(f"Error connecting to server: {e}")
    sys.exit(1) 