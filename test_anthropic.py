"""
test_anthropic.py — isolated test to verify the API key and find a working model.
Run with: .\\venv\\Scripts\\python.exe test_anthropic.py
"""
import os
from pathlib import Path

# Load .env manually (same logic as shot_analysis.py)
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("[OK] python-dotenv loaded .env")
except ImportError:
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
        print("[OK] .env loaded via fallback parser")

api_key = os.environ.get("ANTHROPIC_API_KEY", "")
print(f"\nANTHROPIC_API_KEY found: {'YES' if api_key else 'NO'}")
print(f"Key prefix: {api_key[:20]}..." if api_key else "No key set")

if not api_key:
    print("\nERROR: No API key found. Check .env file.")
    exit(1)

try:
    from anthropic import Anthropic
    print("\n[OK] anthropic package is installed")
except ImportError:
    print("\nERROR: 'anthropic' package not installed. Run: .\\venv\\Scripts\\pip.exe install anthropic")
    exit(1)

client = Anthropic(api_key=api_key)

models_to_try = [
    "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-latest",
    "claude-3-7-sonnet-20250219",
    "claude-3-5-haiku-20241022",
    "claude-3-haiku-20240307",
    "claude-3-sonnet-20240229",
    "claude-3-opus-20240229",
]

print(f"\nTesting {len(models_to_try)} Claude models...\n")
working_model = None
for model in models_to_try:
    try:
        response = client.messages.create(
            model=model,
            max_tokens=50,
            messages=[{"role": "user", "content": "Say OK"}],
        )
        text = response.content[0].text.strip()
        print(f"  [PASS] {model}: '{text}'")
        if working_model is None:
            working_model = model
    except Exception as e:
        print(f"  [FAIL] {model}: {e}")

if working_model:
    print(f"\n==> Best working model: {working_model}")
    print(f"\nUpdate MODEL_CLAUDE in shot_analysis.py to: \"{working_model}\"")
else:
    print("\n==> ALL MODELS FAILED. The API key may be invalid or expired.")
