# hieroglyph_tokenizer.py
# Tokenizer مخصص للرموز الهيروغليفية

import sys
import io
import json
import os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
TOKENIZER_DIR = os.path.dirname(__file__)

# رموز خاصة
PAD_TOKEN = "<PAD>"   # تعبئة
UNK_TOKEN = "<UNK>"   # رمز مجهول
BOS_TOKEN = "<BOS>"   # بداية النص
EOS_TOKEN = "<EOS>"   # نهاية النص
SEP_TOKEN = "<SEP>"   # فاصل بين اللغات

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, SEP_TOKEN]

def build_vocabulary(data):
    """بناء قاموس المفردات من البيانات"""
    vocab = {}
    
    # أولاً الرموز الخاصة
    for i, token in enumerate(SPECIAL_TOKENS):
        vocab[token] = i
    
    next_id = len(SPECIAL_TOKENS)
    
    for word in data:
        # الرموز الهيروغليفية — كل رمز وحدة مستقلة
        hieroglyphs = word.get('hieroglyphs', '')
        for char in hieroglyphs:
            if char not in vocab:
                vocab[char] = next_id
                next_id += 1
        
        # النطق — كل حرف وحدة
        trans = word.get('transliteration', '')
        for char in trans:
            if char not in vocab:
                vocab[char] = next_id
                next_id += 1
        
        # المعنى العربي — كل حرف وحدة
        meaning_ar = word.get('meaning_ar', '')
        for char in meaning_ar:
            if char not in vocab:
                vocab[char] = next_id
                next_id += 1
        
        # المعنى الإنجليزي — كل حرف وحدة
        meaning_en = word.get('meaning_en', '')
        for char in meaning_en:
            if char not in vocab:
                vocab[char] = next_id
                next_id += 1
    
    return vocab

def encode(text, vocab):
    """تحويل النص إلى أرقام"""
    unk_id = vocab[UNK_TOKEN]
    return [vocab.get(char, unk_id) for char in text]

def decode(ids, id_to_token):
    """تحويل الأرقام إلى نص"""
    return ''.join([id_to_token.get(i, UNK_TOKEN) for i in ids])

def save_tokenizer(vocab):
    """حفظ الـ Tokenizer"""
    id_to_token = {v: k for k, v in vocab.items()}
    
    tokenizer_data = {
        "vocab": vocab,
        "id_to_token": {str(k): v for k, v in id_to_token.items()},
        "vocab_size": len(vocab),
        "special_tokens": {
            "pad": vocab[PAD_TOKEN],
            "unk": vocab[UNK_TOKEN],
            "bos": vocab[BOS_TOKEN],
            "eos": vocab[EOS_TOKEN],
            "sep": vocab[SEP_TOKEN],
        }
    }
    
    path = os.path.join(TOKENIZER_DIR, "tokenizer.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)
    
    return path

def main():
    print("=== بناء Tokenizer الهيروغليفي ===\n")
    
    # تحميل البيانات
    data_path = os.path.join(PROCESSED_DIR, "combined_dataset.json")
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"البيانات المحملة: {len(data)} كلمة")
    
    # بناء القاموس
    vocab = build_vocabulary(data)
    print(f"حجم المفردات: {len(vocab)} رمز فريد")
    
    # حفظ الـ Tokenizer
    path = save_tokenizer(vocab)
    print(f"محفوظ: {path}")
    
    # اختبار
    print("\n=== اختبار الـ Tokenizer ===")
    id_to_token = {v: k for k, v in vocab.items()}
    
    test_cases = [
        "𓄿𓏤",
        "vulture",
        "نسر",
        "A"
    ]
    
    for text in test_cases:
        encoded = encode(text, vocab)
        decoded = decode(encoded, id_to_token)
        print(f"النص:     {text}")
        print(f"مشفر:     {encoded}")
        print(f"مفكك:     {decoded}")
        print("---")
    
    # إحصائيات
    hiero_tokens = [k for k in vocab if len(k) == 1 and ord(k) > 0x13000]
    print(f"\nرموز هيروغليفية فريدة: {len(hiero_tokens)}")
    print(f"إجمالي المفردات: {len(vocab)}")

if __name__ == "__main__":
    main()
