# collect_arabic.py
# جمع القاموس العربي من مكتبة الإسكندرية

import sys
import io
import requests
from bs4 import BeautifulSoup
import json
import os
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "raw")
BASE_URL = "https://www.bibalex.org/learnhieroglyphs/Dictionary/List_Ar.aspx"

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

def extract_words(html, phnid, page):
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
            word["meaning_ar"] = texts[0]
        if len(texts) >= 2:
            word["meaning_ar2"] = texts[1]
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
    print("=== جمع القاموس العربي من مكتبة الإسكندرية ===\n")
    all_words = []

    # نجلب الصفحة الأولى لنعرف phnids
    html = fetch_page(f"{BASE_URL}?phnid=1")
    if not html:
        print("❌ فشل الاتصال")
        return

    # كل phnids
    phnids = list(range(1, 25)) + [31]

    for phnid in phnids:
        url = f"{BASE_URL}?phnid={phnid}"
        html = fetch_page(url)
        if not html:
            continue

        total_pages = get_total_pages(html)
        print(f"حرف phnid={phnid} — صفحات: {total_pages}")

        phnid_words = []
        words = extract_words(html, phnid, 1)
        phnid_words.extend(words)
        time.sleep(0.5)

        for page in range(2, total_pages + 1):
            page_url = f"{BASE_URL}?phnid={phnid}&Dir={page}"
            page_html = fetch_page(page_url)
            if page_html:
                words = extract_words(page_html, phnid, page)
                phnid_words.extend(words)
            time.sleep(0.5)

        print(f"  إجمالي: {len(phnid_words)} كلمة")
        all_words.extend(phnid_words)

    save_data(all_words, "dictionary_arabic.json")
    print(f"\n=== انتهى — إجمالي: {len(all_words)} كلمة ===")

if __name__ == "__main__":
    main()
