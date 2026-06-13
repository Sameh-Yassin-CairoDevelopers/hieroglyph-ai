import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
with open('data/raw/dictionary_arabic.json', encoding='utf-8') as f:
    data = json.load(f)
for w in data[:3]:
    print(w)
    print('---')
