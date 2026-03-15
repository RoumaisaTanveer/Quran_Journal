import requests, os

key = os.getenv('OPENROUTER_API_KEY')
if not key:
    print("ERROR: OPENROUTER_API_KEY not set. Run: set OPENROUTER_API_KEY=your_key")
    exit()

res = requests.get('https://openrouter.ai/api/v1/models', headers={'Authorization': f'Bearer {key}'})
models = res.json()['data']
free = [m['id'] for m in models if str(m.get('pricing', {}).get('prompt', '1')) == '0']

print(f"Found {len(free)} free models:\n")
for m in free:
    print(' ', m)
