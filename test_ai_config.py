"""
Test script to verify AI API configuration
"""
import os
from pathlib import Path
import sys

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

# Try to load .env
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"[OK] Loaded .env file from {env_path}")
    else:
        print(f"[WARNING] .env file not found at {env_path}")
        print("   Create a .env file with your API key:")
        print("   GROQ_API_KEY=your_key_here")
        print("   or")
        print("   OPENAI_API_KEY=your_key_here")
        print("   or")
        print("   ANTHROPIC_API_KEY=your_key_here")
except ImportError:
    print("[ERROR] python-dotenv not installed. Run: pip install python-dotenv")
    sys.exit(1)

# Check which API keys are available
print("\n" + "=" * 60)
print("API KEY CONFIGURATION CHECK")
print("=" * 60)

groq_key = os.environ.get("GROQ_API_KEY")
openai_key = os.environ.get("OPENAI_API_KEY")
anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
ai_model = os.environ.get("AI_MODEL", "gpt-4")

print(f"\nGROQ_API_KEY: {'[SET]' if groq_key else '[NOT SET]'}")
if groq_key:
    print(f"  Key: {groq_key[:10]}...{groq_key[-4:] if len(groq_key) > 14 else '***'}")

print(f"\nOPENAI_API_KEY: {'[SET]' if openai_key else '[NOT SET]'}")
if openai_key:
    print(f"  Key: {openai_key[:10]}...{openai_key[-4:] if len(openai_key) > 14 else '***'}")

print(f"\nANTHROPIC_API_KEY: {'[SET]' if anthropic_key else '[NOT SET]'}")
if anthropic_key:
    print(f"  Key: {anthropic_key[:10]}...{anthropic_key[-4:] if len(anthropic_key) > 14 else '***'}")

print(f"\nAI_MODEL: {ai_model}")

# Determine which API will be used
print("\n" + "=" * 60)
print("API PRIORITY (which will be used)")
print("=" * 60)

if groq_key:
    print("\n[PRIORITY 1] Groq (kostenlos, schnell)")
    print("  Will use: Groq API")
    print("  Model: llama-3.1-70b-versatile (or AI_MODEL if specified)")
    print("  Get API key: https://console.groq.com/")
elif openai_key:
    print("\n[PRIORITY 2] OpenAI (kostenpflichtig)")
    print("  Will use: OpenAI API")
    print(f"  Model: {ai_model}")
    print("  Get API key: https://platform.openai.com/api-keys")
elif anthropic_key:
    print("\n[PRIORITY 3] Anthropic Claude (kostenpflichtig)")
    print("  Will use: Anthropic API")
    print(f"  Model: {ai_model}")
    print("  Get API key: https://console.anthropic.com/")
else:
    print("\n[WARNING] No API key found!")
    print("  AI scores will use heuristic fallback (no LLM)")
    print("\n  To use LLM:")
    print("  1. Get a free Groq API key: https://console.groq.com/")
    print("  2. Add to .env file: GROQ_API_KEY=your_key_here")
    print("  3. Or use OpenAI/Anthropic (costs money)")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)

