# collect_data.py
# استخراج كامل قاموس الهيروغليفية مع كل الصفحات

import sys
import io
import requests
from bs4 import BeautifulSoup
import json
import os
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "raw")
BASE_URL = "https://www.bibalex.org/learnhieroglyphs/Dictionary/List_En.aspx"

def fetch_page(url):
    headers = {"User-Agent": "Mozilla/5.0 (hieroglyph-ai research project)"}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"  خطأ: {e}")
        return None

def get_total_pages(html):
    """استخراج عدد الصفحات الكلي"""
    soup = BeautifulSoup(html, "html.parser")
    max_dir = 1
    for a in soup.find_all('a'):
        href = a.get('href', '')
        if 'Dir=' in href:
            try:
                dir_val = int(href.split('Dir=')[1])
                if dir_val > max_dir:
                    max_dir = dir_val
            except:
                pass
    return max_dir

def get_all_phnids(html):
    """استخراج كل phnid الموجودة في القائمة"""
    soup = BeautifulSoup(html, "html.parser")
    phnids = set()
    for a in soup.find_all('a'):
        href = a.get('href', '')
        if 'phnid=' in href and 'Dir=' not in href:
            try:
                phnid = int(href.split('phnid=')[1])
                phnids.add(phnid)
            except:
                pass
    return sorted(phnids)

def extract_words(html, phnid, page):
    """استخراج الكلمات من الصفحة"""
    soup = BeautifulSoup(html, "html.parser")
    words = []

    cards = soup.find_all("div", class_="service-block")

    for card in cards:
        word = {}

        trans = card.find("div", class_="transliteration")
        if trans:
            word["transliteration"] = trans.get_text(strip=True)

        plain_divs = card.find_all("div", class_=False)
        texts = [d.get_text(strip=True).replace('\xa0', ' ') for d in plain_divs if d.get_text(strip=True)]

        if len(texts) >= 1:
            word["meaning_en"] = texts[0]
        if len(texts) >= 2:
            word["meaning_ar"] = texts[1]

        inner = card.find("div", class_="inner-box")
        if inner:
            word["hieroglyphs_raw"] = inner.get_text(strip=True)

        if word:
            word["phnid"] = phnid
            word["page"] = page
            words.append(word)

    return words

def save_data(data, filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  محفوظ: {filepath}")

def main():
    print("=== بدء جمع القاموس الهيروغليفي الكامل ===\n")

    # أولاً نجلب الصفحة الرئيسية لنعرف كل phnids
    html = fetch_page(f"{BASE_URL}?phnid=1")
    phnids = get_all_phnids(html)
    phnids = sorted(set(phnids + list(range(1, 25))))
    print(f"الأحرف الموجودة: {phnids}\n")

    all_words = []

    for phnid in phnids:
        # نجلب الصفحة الأولى لنعرف عدد الصفحات
        url = f"{BASE_URL}?phnid={phnid}"
        html = fetch_page(url)
        if not html:
            continue

        total_pages = get_total_pages(html)
        print(f"حرف phnid={phnid} — عدد الصفحات: {total_pages}")

        phnid_words = []

        # الصفحة الأولى
        words = extract_words(html, phnid, 1)
        phnid_words.extend(words)
        time.sleep(0.5)

        # باقي الصفحات
        for page in range(2, total_pages + 1):
            page_url = f"{BASE_URL}?phnid={phnid}&Dir={page}"
            page_html = fetch_page(page_url)
            if page_html:
                words = extract_words(page_html, phnid, page)
                phnid_words.extend(words)
                print(f"  صفحة {page}: {len(words)} كلمة")
            time.sleep(0.5)

        print(f"  إجمالي phnid={phnid}: {len(phnid_words)} كلمة")
        all_words.extend(phnid_words)
        save_data(phnid_words, f"phnid_{phnid}.json")

    save_data(all_words, "dictionary_complete.json")
    print(f"\n=== انتهى — إجمالي الكلمات: {len(all_words)} ===")

if __name__ == "__main__":
    main()
