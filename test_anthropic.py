import os
from dotenv import load_dotenv
import anthropic

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")

if not api_key:
    print("❌ ANTHROPIC_API_KEY not found in .env file")
    exit(1)

print(f"✓ API key found: {api_key[:15]}...")

# Test the API
client = anthropic.Anthropic(api_key=api_key)

print("\n🧪 Testing different Claude models...\n")

models_to_test = [
    "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-20240620", 
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307"
]

working_model = None

for model in models_to_test:
    try:
        print(f"Testing: {model}...", end=" ")
        response = client.messages.create(
            model=model,
            max_tokens=10,
            messages=[{"role": "user", "content": "Hi"}]
        )
        print(f"✅ WORKS!")
        working_model = model
        break
    except anthropic.NotFoundError:
        print(f"❌ Not found")
    except anthropic.AuthenticationError:
        print(f"❌ Authentication failed - check your API key")
        break
    except anthropic.PermissionDeniedError:
        print(f"❌ Permission denied - billing may not be set up")
        break
    except Exception as e:
        print(f"❌ Error: {str(e)[:50]}")

if working_model:
    print(f"\n✅ SUCCESS! Working model: {working_model}")
    print(f"\nUpdate your agents.py with:")
    print(f'    model="{working_model}",')
else:
    print("\n❌ NO MODELS WORK!")
    print("\nPossible issues:")
    print("1. Billing not set up in Anthropic console")
    print("2. API key doesn't have permissions")
    print("3. Account not activated yet")
    print("\nSteps to fix:")
    print("1. Go to: https://console.anthropic.com/settings/billing")
    print("2. Add payment method")
    print("3. Verify billing is active")
    print("4. Wait 5 minutes")
    print("5. Run this test again")