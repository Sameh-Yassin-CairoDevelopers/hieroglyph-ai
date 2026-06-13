 
# merge_all_datasets.py
# دمج كل مصادر البيانات في dataset واحد

import json
import os
import re

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RAW_DIR  = os.path.join(BASE_DIR, "data", "raw")
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")

all_data = []

# 1 — مكتبة الإسكندرية
with open(os.path.join(PROC_DIR, "dictionary_clean.json"), encoding="utf-8") as f:
    ba = json.load(f)
for item in ba:
    if item.get("hieroglyphs") and (item.get("meaning_ar") or item.get("meaning_en")):
        all_data.append({
            "hieroglyphs":     item["hieroglyphs"],
            "transliteration": item.get("transliteration", ""),
            "meaning_ar":      item.get("meaning_ar", ""),
            "meaning_en":      item.get("meaning_en", ""),
            "meaning_de":      "",
            "source":          "bibliotheca_alexandrina"
        })
print(f"مكتبة الإسكندرية:  {len([d for d in all_data if d['source']=='bibliotheca_alexandrina'])}")

# 2 — PhilologEg
with open(os.path.join(PROC_DIR, "philologeg_clean.json"), encoding="utf-8") as f:
    philo = json.load(f)
for item in philo:
    if item.get("hieroglyphs") and item.get("transliteration"):
        all_data.append({
            "hieroglyphs":     item["hieroglyphs"],
            "transliteration": item["transliteration"],
            "meaning_ar":      "",
            "meaning_en":      item["transliteration"],
            "meaning_de":      "",
            "source":          "philologeg"
        })
print(f"PhilologEg:         {len([d for d in all_data if d['source']=='philologeg'])}")

# 3 — TLA Earlier Egyptian
with open(os.path.join(RAW_DIR, "tla_corpus.json"), encoding="utf-8") as f:
    tla = json.load(f)
for item in tla:
    if item.get("hieroglyphs") and item.get("transliteration"):
        all_data.append({
            "hieroglyphs":     item["hieroglyphs"],
            "transliteration": item["transliteration"],
            "meaning_ar":      "",
            "meaning_en":      "",
            "meaning_de":      item.get("meaning_de", ""),
            "source":          "tla_earlier"
        })
print(f"TLA Earlier:        {len([d for d in all_data if d['source']=='tla_earlier'])}")

# 4 — TLA Demotic (نطق فقط بدون هيروغليفية)
with open(os.path.join(RAW_DIR, "tla_demotic.json"), encoding="utf-8") as f:
    demotic = json.load(f)
for item in demotic:
    if item.get("transliteration") and item.get("meaning_de"):
        all_data.append({
            "hieroglyphs":     "",
            "transliteration": item["transliteration"],
            "meaning_ar":      "",
            "meaning_en":      "",
            "meaning_de":      item["meaning_de"],
            "source":          "tla_demotic"
        })
print(f"TLA Demotic:        {len([d for d in all_data if d['source']=='tla_demotic'])}")

# إزالة المكررات بناءً على الهيروغليفية
seen = set()
unique = []
for item in all_data:
    key = item["hieroglyphs"] + item["transliteration"]
    if key not in seen:
        seen.add(key)
        unique.append(item)

print(f"\nقبل إزالة المكررات: {len(all_data)}")
print(f"بعد إزالة المكررات: {len(unique)}")

# إحصائيات
has_hiero  = sum(1 for d in unique if d["hieroglyphs"])
has_ar     = sum(1 for d in unique if d["meaning_ar"])
has_en     = sum(1 for d in unique if d["meaning_en"])
has_de     = sum(1 for d in unique if d["meaning_de"])

print(f"\nفيها هيروغليفية:  {has_hiero}")
print(f"فيها عربي:        {has_ar}")
print(f"فيها إنجليزي:     {has_en}")
print(f"فيها ألماني:      {has_de}")

# حفظ
os.makedirs(PROC_DIR, exist_ok=True)
out = os.path.join(PROC_DIR, "master_dataset.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(unique, f, ensure_ascii=False, indent=2)

print(f"\n✅ Master Dataset محفوظ: {len(unique)} عينة")
print(f"المسار: {out}")
