# fetch_openglyph.py
# جمع بيانات OpenGlyph من مصدر نصي مباشر

import requests
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RAW_DIR  = os.path.join(BASE_DIR, "data", "raw")

# نجرب رابط الـ Lexicon مباشرة
urls = [
    "https://huggingface.co/datasets/HamdiJr/Egyptian_hieroglyphs/resolve/main/Dataset/LanguageModel/Lexicon.txt",
    "https://raw.githubusercontent.com/OpenGlyph/OpenGlyph/main/lexicon.txt",
    "https://raw.githubusercontent.com/HamdiJr/Egyptian_hieroglyphs/main/Dataset/LanguageModel/Lexicon.txt",
]

headers = {"User-Agent": "hieroglyph-ai research project"}

for url in urls:
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code == 200:
            print(f"✅ وجدنا: {url}")
            print(f"الحجم: {len(r.text)} حرف")
            print("\nأول 3 أسطر:")
            for line in r.text.split('\n')[:3]:
                print(line)
            
            # حفظ
            os.makedirs(RAW_DIR, exist_ok=True)
            with open(os.path.join(RAW_DIR, "lexicon.txt"), "w", encoding="utf-8") as f:
                f.write(r.text)
            print("\n✅ محفوظ")
            break
        else:
            print(f"❌ {r.status_code} — {url}")
    except Exception as e:
        print(f"❌ {e}")
