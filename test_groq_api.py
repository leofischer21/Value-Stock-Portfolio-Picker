"""
Simple test to verify Groq API works
"""
import os
import sys
from pathlib import Path

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except:
    pass

groq_key = os.environ.get("GROQ_API_KEY")
print(f"GROQ_API_KEY: {'SET' if groq_key else 'NOT SET'}")

if not groq_key:
    print("ERROR: No Groq API key found in environment")
    sys.exit(1)

try:
    from groq import Groq
    print("Groq package: OK")
    
    client = Groq(api_key=groq_key)
    print("Groq client: Created")
    
    # Simple test
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Respond only with valid JSON."},
            {"role": "user", "content": 'Respond with JSON: {"test": "success", "value": 1.0}'}
        ],
        temperature=0.3,
        max_tokens=100
    )
    
    result = response.choices[0].message.content
    print(f"Groq API Response: {result}")
    print("\n[SUCCESS] Groq API is working!")
    
except ImportError as e:
    print(f"ERROR: groq package not installed: {e}")
    print("Install with: pip install groq")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: Groq API call failed: {e}")
    sys.exit(1)

