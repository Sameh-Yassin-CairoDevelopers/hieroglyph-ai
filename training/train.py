# train.py
# تدريب النموذج على القاموس الهيروغليفي

import sys
import json
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random


# المسارات
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
TOKENIZER_DIR = os.path.join(BASE_DIR, "tokenizer")
MODEL_DIR = os.path.join(BASE_DIR, "model")
TRAINING_DIR = os.path.dirname(__file__)

sys.path.append(BASE_DIR)
from model.transformer import HieroglyphTransformer


class HieroglyphDataset(Dataset):
    """Dataset للهيروغليفية"""

    def __init__(self, data, vocab, max_len=64):
        self.samples = []
        self.vocab = vocab
        self.max_len = max_len

        bos = vocab["<BOS>"]
        eos = vocab["<EOS>"]
        pad = vocab["<PAD>"]
        unk = vocab["<UNK>"]

        for word in data:
            hieroglyphs = word.get("hieroglyphs", "")
            meaning_ar = word.get("meaning_ar", "")

            # نتجاهز الكلمات بدون رموز هيروغليفية
            if not hieroglyphs or not meaning_ar:
                continue

            # تشفير المدخل (هيروغليفية)
            src = [bos] + [vocab.get(c, unk) for c in hieroglyphs] + [eos]
            # تشفير المخرج (عربي)
            tgt = [bos] + [vocab.get(c, unk) for c in meaning_ar] + [eos]

            # تقليم أو تعبئة
            src = src[:max_len]
            tgt = tgt[:max_len]

            self.samples.append((src, tgt))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """تجميع العينات مع Padding"""
    src_batch, tgt_batch = zip(*batch)

    src_lens = [len(s) for s in src_batch]
    tgt_lens = [len(t) for t in tgt_batch]

    max_src = max(src_lens)
    max_tgt = max(tgt_lens)

    src_padded = [s + [0] * (max_src - len(s)) for s in src_batch]
    tgt_padded = [t + [0] * (max_tgt - len(t)) for t in tgt_batch]

    src_tensor = torch.tensor(src_padded, dtype=torch.long)
    tgt_tensor = torch.tensor(tgt_padded, dtype=torch.long)

    return src_tensor, tgt_tensor


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        optimizer.zero_grad()
        output = model(src, tgt_input)

        output = output.reshape(-1, output.size(-1))
        tgt_output = tgt_output.reshape(-1)

        loss = criterion(output, tgt_output)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def main():
    print("=== بدء التدريب ===\n")

    # تحميل البيانات
    with open(os.path.join(PROCESSED_DIR, "dictionary_clean.json"), encoding="utf-8") as f:
        data = json.load(f)

    # تحميل الـ Tokenizer
    with open(os.path.join(TOKENIZER_DIR, "tokenizer.json"), encoding="utf-8") as f:
        tokenizer = json.load(f)

    vocab = tokenizer["vocab"]
    vocab_size = tokenizer["vocab_size"]
    print(f"البيانات: {len(data)} كلمة")
    print(f"المفردات: {vocab_size} رمز")

    # تقسيم البيانات
    random.shuffle(data)
    split = int(len(data) * 0.9)
    train_data = data[:split]
    val_data = data[split:]
    print(f"تدريب: {len(train_data)} | تحقق: {len(val_data)}")

    # إنشاء Datasets
    train_dataset = HieroglyphDataset(train_data, vocab)
    val_dataset = HieroglyphDataset(val_data, vocab)
    print(f"عينات التدريب: {len(train_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=32,
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32,
                            shuffle=False, collate_fn=collate_fn)

    # الجهاز
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"الجهاز: {device}\n")

    # النموذج
    model = HieroglyphTransformer(vocab_size=vocab_size).to(device)
    print(f"حجم النموذج: {model.count_parameters():,} parameter")

    # التدريب
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_loss = float("inf")
    epochs = 20

    print("\nالـ Epoch | خسارة التدريب | خسارة التحقق")
    print("-" * 45)

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # التحقق
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                output = model(src, tgt_input)
                output = output.reshape(-1, output.size(-1))
                tgt_output = tgt_output.reshape(-1)
                val_loss += criterion(output, tgt_output).item()
        val_loss /= len(val_loader)

        scheduler.step()

        print(f"  {epoch:4d}   |    {train_loss:.4f}      |    {val_loss:.4f}")

        # حفظ أفضل نموذج
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(),
                       os.path.join(TRAINING_DIR, "best_model.pt"))
            print(f"         ✅ حفظ أفضل نموذج")

    print(f"\n=== انتهى التدريب — أفضل خسارة: {best_loss:.4f} ===")


if __name__ == "__main__":
    main()
