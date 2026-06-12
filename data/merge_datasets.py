 
# merge_datasets.py
# دمج كل مصادر البيانات في dataset واحد للتدريب

import sys
import json
import os
import re

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

def main():
    print("=== دمج مصادر البيانات ===\n")

    all_data = []

    # المصدر 1 — قاموس مكتبة الإسكندرية
    path1 = os.path.join(PROCESSED_DIR, "dictionary_clean.json")
    with open(path1, encoding="utf-8") as f:
        dict_data = json.load(f)

    for item in dict_data:
        hiero = item.get("hieroglyphs", "")
        trans = item.get("transliteration", "")
        ar    = item.get("meaning_ar", "")
        en    = item.get("meaning_en", "")

        if hiero and ar:
            all_data.append({
                "hieroglyphs":     hiero,
                "transliteration": trans,
                "meaning_ar":      ar,
                "meaning_en":      en,
                "source":          "bibliotheca_alexandrina"
            })

    print(f"مكتبة الإسكندرية: {len(dict_data)} كلمة")

    # المصدر 2 — PhilologEg
    path2 = os.path.join(PROCESSED_DIR, "philologeg_clean.json")
    with open(path2, encoding="utf-8") as f:
        philo_data = json.load(f)

    for item in philo_data:
        hiero = item.get("hieroglyphs", "")
        trans = item.get("transliteration", "")

        if hiero and trans:
            all_data.append({
                "hieroglyphs":     hiero,
                "transliteration": trans,
                "meaning_ar":      "",
                "meaning_en":      trans,
                "source":          "philologeg"
            })

    print(f"PhilologEg:        {len(philo_data)} جملة")

    # إزالة المكررات
    seen = set()
    unique = []
    for item in all_data:
        key = item["hieroglyphs"]
        if key not in seen:
            seen.add(key)
            unique.append(item)

    print(f"\nقبل إزالة المكررات: {len(all_data)}")
    print(f"بعد إزالة المكررات: {len(unique)}")

    # حفظ
    output = os.path.join(PROCESSED_DIR, "combined_dataset.json")
    with open(output, "w", encoding="utf-8") as f:
        json.dump(unique, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Dataset مدموج محفوظ: {len(unique)} عينة")
    print(f"المسار: {output}")

if __name__ == "__main__":
    main()
