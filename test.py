import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

AZURE_OPENAI_GPT4_ENDPOINT = os.getenv("AZURE_OPENAI_GPT4_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_API_KEY")

# Test API Connection
headers = {
    "Content-Type": "application/json",
    "api-key": AZURE_API_KEY
}

test_payload = {
    "messages": [{"role": "system", "content": "Say hello!"}],
    "max_tokens": 10
}

response = requests.post(AZURE_OPENAI_GPT4_ENDPOINT, headers=headers, json=test_payload)
print(response.json())
