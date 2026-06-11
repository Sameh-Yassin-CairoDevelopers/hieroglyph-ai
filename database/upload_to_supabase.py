 
# upload_to_supabase.py
# رفع بيانات القاموس لـ Supabase

import sys
import json
import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

def main():
    print("=== رفع البيانات لـ Supabase ===\n")

    # الاتصال
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("✅ تم الاتصال بـ Supabase")

    # تحميل البيانات
    with open(os.path.join(PROCESSED_DIR, "dictionary_clean.json"), encoding="utf-8") as f:
        data = json.load(f)
    print(f"البيانات المحملة: {len(data)} كلمة")

    # رفع على دفعات
    batch_size = 100
    total = 0

    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]

        rows = [{
            "transliteration": w.get("transliteration", ""),
            "hieroglyphs":     w.get("hieroglyphs", ""),
            "meaning_ar":      w.get("meaning_ar", ""),
            "meaning_en":      w.get("meaning_en", ""),
            "phnid":           w.get("phnid", 0)
        } for w in batch]

        supabase.table("words").insert(rows).execute()
        total += len(rows)
        print(f"  رُفع: {total}/{len(data)}")

    # حفظ نتيجة التدريب
    supabase.table("training_sessions").insert({
        "epochs":       20,
        "best_loss":    2.3861,
        "train_loss":   2.3644,
        "val_loss":     2.3861,
        "model_version": "v1.0",
        "total_words":  len(data),
        "notes":        "First training - Bibliotheca Alexandrina dataset"
    }).execute()
    print("\n✅ حُفظت نتيجة التدريب")

    print(f"\n=== انتهى — {total} كلمة في Supabase ===")

if __name__ == "__main__":
    main()
