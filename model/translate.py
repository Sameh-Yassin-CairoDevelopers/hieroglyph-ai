 
# translate.py
# اختبار النموذج — ترجمة الهيروغليفية

import sys
import json
import os
import torch

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
TOKENIZER_DIR = os.path.join(BASE_DIR, "tokenizer")
TRAINING_DIR = os.path.join(BASE_DIR, "training")
MODEL_DIR = os.path.dirname(__file__)

sys.path.append(BASE_DIR)
from model.transformer import HieroglyphTransformer

def load_tokenizer():
    with open(os.path.join(TOKENIZER_DIR, "tokenizer.json"), encoding="utf-8") as f:
        return json.load(f)

def translate(hieroglyphs, model, vocab, id_to_token, max_len=50):
    """ترجمة رموز هيروغليفية إلى عربي"""
    model.eval()
    
    bos = vocab["<BOS>"]
    eos = vocab["<EOS>"]
    unk = vocab["<UNK>"]
    
    # تشفير المدخل
    src = [bos] + [vocab.get(c, unk) for c in hieroglyphs] + [eos]
    src_tensor = torch.tensor([src], dtype=torch.long)
    
    # توليد الترجمة
    tgt = [bos]
    
    with torch.no_grad():
        for _ in range(max_len):
            tgt_tensor = torch.tensor([tgt], dtype=torch.long)
            output = model(src_tensor, tgt_tensor)
            logits = output[0, -1, :]
            # عقوبة التكرار
            for token_id in set(tgt):
                logits[token_id] -= 2.0
            next_token = logits.argmax().item()
            
            if next_token == eos:
                break
            tgt.append(next_token)
    
    # فك التشفير
    result = ''.join([id_to_token.get(str(i), '') for i in tgt[1:]])
    return result

def main():
    print("=== اختبار ترجمة الهيروغليفية ===\n")
    
    # تحميل الـ Tokenizer
    tokenizer = load_tokenizer()
    vocab = tokenizer["vocab"]
    id_to_token = tokenizer["id_to_token"]
    vocab_size = tokenizer["vocab_size"]
    
    # تحميل النموذج
    model = HieroglyphTransformer(vocab_size=vocab_size)
    model_path = os.path.join(TRAINING_DIR, "best_model.pt")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    print("✅ النموذج محمّل\n")
    
    # اختبارات
    test_cases = [
        "𓄿𓏤",        # نسر، طائر
        "𓄿",          # A
        "𓄿𓇋𓏲𓊡",    # عاصف
        "𓄿𓄿𓈇𓏤",    # تل من الأطلال
    ]
    
    print("الهيروغليفية  →  الترجمة")
    print("-" * 40)
    
    for hiero in test_cases:
        result = translate(hiero, model, vocab, id_to_token)
        print(f"{hiero}  →  {result}")

if __name__ == "__main__":
    main()
