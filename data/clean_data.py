# clean_data.py
# تنظيف بيانات القاموس الهيروغليفي وتجهيزها للتدريب

import sys
import io
import json
import os
import re
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "processed")

def load_complete_dictionary():
    """تحميل القاموس الكامل"""
    filepath = os.path.join(RAW_DIR, "dictionary_complete.json")
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)

def clean_text(text):
    """تنظيف النص من المسافات والرموز غير المرغوبة"""
    if not text:
        return ""
    # إزالة المسافات غير العادية
    text = text.replace('\xa0', ' ')
    text = text.replace('\u200b', '')
    # تنظيف المسافات المتعددة
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_hieroglyphs(raw_text, transliteration):
    """استخراج الرموز الهيروغليفية فقط من النص الخام"""
    if not raw_text:
        return ""
    # إزالة النطق من البداية
    text = raw_text
    if transliteration and raw_text.startswith(transliteration):
        text = raw_text[len(transliteration):]
    # استخراج الرموز الهيروغليفية (Unicode range)
    hieroglyphs = re.findall(r'[\U00013000-\U0001342F\U00013430-\U0001343F]+', text)
    return ''.join(hieroglyphs)

def clean_word(word):
    """تنظيف كلمة واحدة"""
    cleaned = {}

    cleaned['transliteration'] = clean_text(word.get('transliteration', ''))
    cleaned['meaning_en'] = clean_text(word.get('meaning_en', ''))
    cleaned['meaning_ar'] = clean_text(word.get('meaning_ar', ''))
    cleaned['phnid'] = word.get('phnid', 0)

    # استخراج الرموز الهيروغليفية
    raw = word.get('hieroglyphs_raw', '')
    cleaned['hieroglyphs'] = extract_hieroglyphs(raw, cleaned['transliteration'])

    return cleaned

def is_valid(word):
    """التحقق من أن الكلمة مكتملة وصالحة"""
    has_trans = bool(word.get('transliteration'))
    has_meaning = bool(word.get('meaning_en') or word.get('meaning_ar'))
    return has_trans and has_meaning

def main():
    print("=== تنظيف البيانات ===\n")

    # تحميل البيانات الخام
    raw_data = load_complete_dictionary()
    print(f"البيانات الخام: {len(raw_data)} كلمة")

    # تنظيف كل كلمة
    cleaned = [clean_word(w) for w in raw_data]

    # فلترة الكلمات غير المكتملة
    valid = [w for w in cleaned if is_valid(w)]
    print(f"بعد التنظيف: {len(valid)} كلمة صالحة")
    print(f"تم حذف: {len(cleaned) - len(valid)} كلمة ناقصة")

    # حفظ JSON نظيف
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    json_path = os.path.join(PROCESSED_DIR, "dictionary_clean.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(valid, f, ensure_ascii=False, indent=2)
    print(f"\nمحفوظ: {json_path}")

    # حفظ CSV للمراجعة البشرية
    df = pd.DataFrame(valid)
    csv_path = os.path.join(PROCESSED_DIR, "dictionary_clean.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"محفوظ: {csv_path}")

    # إحصائيات
    print(f"\n=== إحصائيات ===")
    print(f"كلمات فيها رموز هيروغليفية: {df['hieroglyphs'].astype(bool).sum()}")
    print(f"كلمات فيها معنى عربي: {df['meaning_ar'].astype(bool).sum()}")
    print(f"كلمات فيها معنى إنجليزي: {df['meaning_en'].astype(bool).sum()}")

    # عرض 3 أمثلة
    print(f"\n=== أمثلة ===")
    for w in valid[:3]:
        print(w)
        print("---")

if __name__ == "__main__":
    main()
