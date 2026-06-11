# train.py
# تدريب النموذج على القاموس الهيروغليفي — النسخة المحسّنة

import sys
import json
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
TOKENIZER_DIR = os.path.join(BASE_DIR, "tokenizer")
MODEL_DIR = os.path.join(BASE_DIR, "model")
TRAINING_DIR = os.path.dirname(__file__)

sys.path.append(BASE_DIR)
from model.transformer import HieroglyphTransformer


class HieroglyphDataset(Dataset):

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
            meaning_ar  = word.get("meaning_ar", "")

            if not hieroglyphs or not meaning_ar:
                continue

            src = [bos] + [vocab.get(c, unk) for c in hieroglyphs] + [eos]
            tgt = [bos] + [vocab.get(c, unk) for c in meaning_ar]  + [eos]

            src = src[:max_len]
            tgt = tgt[:max_len]

            self.samples.append((src, tgt))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)

    max_src = max(len(s) for s in src_batch)
    max_tgt = max(len(t) for t in tgt_batch)

    src_padded = [s + [0] * (max_src - len(s)) for s in src_batch]
    tgt_padded = [t + [0] * (max_tgt - len(t)) for t in tgt_batch]

    return (
        torch.tensor(src_padded, dtype=torch.long),
        torch.tensor(tgt_padded, dtype=torch.long)
    )


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)

        tgt_input  = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        optimizer.zero_grad()
        output = model(src, tgt_input)

        loss = criterion(
            output.reshape(-1, output.size(-1)),
            tgt_output.reshape(-1)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)

            tgt_input  = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            output = model(src, tgt_input)
            loss = criterion(
                output.reshape(-1, output.size(-1)),
                tgt_output.reshape(-1)
            )
            total_loss += loss.item()

    return total_loss / len(loader)


def main():
    print("=== بدء التدريب المحسّن ===\n")

    # تحميل البيانات
    with open(os.path.join(PROCESSED_DIR, "dictionary_clean.json"), encoding="utf-8") as f:
        data = json.load(f)

    with open(os.path.join(TOKENIZER_DIR, "tokenizer.json"), encoding="utf-8") as f:
        tokenizer = json.load(f)

    vocab      = tokenizer["vocab"]
    vocab_size = tokenizer["vocab_size"]

    print(f"البيانات:  {len(data)} كلمة")
    print(f"المفردات:  {vocab_size} رمز")

    # تقسيم البيانات
    random.seed(42)
    random.shuffle(data)
    split      = int(len(data) * 0.9)
    train_data = data[:split]
    val_data   = data[split:]

    train_dataset = HieroglyphDataset(train_data, vocab)
    val_dataset   = HieroglyphDataset(val_data,   vocab)

    print(f"تدريب: {len(train_dataset)} | تحقق: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, collate_fn=collate_fn)

    # الجهاز
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"الجهاز:    {device}\n")

    # النموذج
    model = HieroglyphTransformer(vocab_size=vocab_size).to(device)

    # استكمال التدريب من آخر نقطة
    checkpoint = os.path.join(TRAINING_DIR, "best_model.pt")
    if os.path.exists(checkpoint):
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        print("✅ تم تحميل النموذج السابق — نكمل من loss 2.38\n")

    print(f"حجم النموذج: {model.count_parameters():,} parameter")

    # إعداد التدريب
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_loss  = float("inf")
    epochs     = 40
    no_improve = 0

    print("الـ Epoch | خسارة التدريب | خسارة التحقق | أفضل")
    print("-" * 55)

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss   = eval_epoch(model,   val_loader,  criterion, device)

        scheduler.step(val_loss)

        improved = ""
        if val_loss < best_loss:
            best_loss  = val_loss
            no_improve = 0
            torch.save(model.state_dict(), checkpoint)
            improved = "✅"
        else:
            no_improve += 1

        print(f"  {epoch:4d}   |    {train_loss:.4f}      |    {val_loss:.4f}    | {improved}")

        # Early Stopping — نوقف لو لم يتحسن 10 epochs متتالية
        if no_improve >= 10:
            print(f"\n⏹ Early Stopping — لم يتحسن منذ {no_improve} epochs")
            break

    print(f"\n=== انتهى التدريب — أفضل خسارة: {best_loss:.4f} ===")


if __name__ == "__main__":
    main()
