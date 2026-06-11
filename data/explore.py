# inspect.py - فحص بنية HTML الصفحة
import sys
import io
import requests
from bs4 import BeautifulSoup

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

r = requests.get('https://www.bibalex.org/learnhieroglyphs/Dictionary/List_En.aspx?phnid=1')
soup = BeautifulSoup(r.text, 'html.parser')

divs = soup.find_all('div')
count = 0
for d in divs:
    text = d.get_text(strip=True)[:60]
    cls = d.get('class')
    if text:
        print(f"CLASS: {cls} --- TEXT: {text}")
        count += 1
    if count >= 50:
        break
