# clean_arabic.py
import json, os, re, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RAW_DIR  = os.path.join(BASE_DIR, "data", "raw")
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")

def extract_hieroglyphs(raw_text, transliteration):
    if not raw_text:
        return ""
    text = raw_text
    if transliteration and raw_text.startswith(transliteration):
        text = raw_text[len(transliteration):]
    hieroglyphs = re.findall(r'[\U00013000-\U0001342F\U00013430-\U0001343F]+', text)
    return ''.join(hieroglyphs)

def clean_text(text):
    if not text:
        return ""
    text = text.replace('\xa0', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def main():
    print("=== تنظيف القاموس العربي ===\n")

    with open(os.path.join(RAW_DIR, "dictionary_arabic.json"), encoding="utf-8") as f:
        data = json.load(f)

    print(f"البيانات الخام: {len(data)}")

    cleaned = []
    for item in data:
        trans  = clean_text(item.get("transliteration", ""))
        ar     = clean_text(item.get("meaning_ar", ""))
        en     = clean_text(item.get("meaning_ar2", ""))
        raw    = item.get("hieroglyphs_raw", "")
        hiero  = extract_hieroglyphs(raw, trans)

        if trans and (ar or en):
            cleaned.append({
                "hieroglyphs":     hiero,
                "transliteration": trans,
                "meaning_ar":      ar,
                "meaning_en":      en,
                "meaning_de":      "",
                "source":          "bibliotheca_alexandrina_ar"
            })

    print(f"بعد التنظيف: {len(cleaned)}")
    print(f"فيها هيروغليفية: {sum(1 for w in cleaned if w['hieroglyphs'])}")
    print(f"فيها عربي:       {sum(1 for w in cleaned if w['meaning_ar'])}")
    print(f"فيها إنجليزي:    {sum(1 for w in cleaned if w['meaning_en'])}")

    # أمثلة
    print("\n=== أمثلة ===")
    for w in cleaned[:3]:
        print(w)
        print("---")

    # حفظ
    os.makedirs(PROC_DIR, exist_ok=True)
    out = os.path.join(PROC_DIR, "dictionary_arabic_clean.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)
    print(f"\n✅ محفوظ: {out}")

if __name__ == "__main__":
    main()
