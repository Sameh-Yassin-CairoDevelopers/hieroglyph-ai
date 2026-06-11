# collect_data.py
# الهدف: جمع الرموز الهيروغليفية من مكتبة الإسكندرية

import requests
from bs4 import BeautifulSoup
import json
import os
import time

# المجلد الذي سنحفظ فيه البيانات
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "raw")

def fetch_page(url):
    """تحميل صفحة ويب والتعامل مع الأخطاء ببساطة"""
    headers = {
        "User-Agent": "Mozilla/5.0 (hieroglyph-ai research project)"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"خطأ في تحميل الصفحة: {e}")
        return None

def parse_hieroglyph_entry(html):
    """استخراج الرموز ومعانيها من الصفحة"""
    soup = BeautifulSoup(html, "html.parser")
    entries = []

    # طباعة عنوان الصفحة للتأكد أننا وصلنا للموقع
    title = soup.find("title")
    if title:
        print(f"الصفحة: {title.text.strip()}")

    return entries

def save_data(data, filename):
    """حفظ البيانات كملف JSON"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"تم الحفظ: {filepath}")

def main():
    print("=== بدء جمع البيانات ===")
    
    # الموقع الرئيسي
    url = "https://www.bibalex.org/learnhieroglyphs/Home/index_en.aspx"
    
    print(f"جاري الاتصال بـ {url}")
    html = fetch_page(url)
    
    if html:
        print("✅ تم الاتصال بالموقع بنجاح")
        entries = parse_hieroglyph_entry(html)
        
        # حفظ نتيجة أولية للتأكد
        save_data({"status": "connected", "url": url}, "test_connection.json")
    else:
        print("❌ فشل الاتصال")

    print("=== انتهى ===")

if __name__ == "__main__":
    main()
