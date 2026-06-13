# translate.py
import sys
import json
import os
import torch
import torch.nn as nn
import math

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
TOKENIZER_DIR = os.path.join(BASE_DIR, "tokenizer")
TRAINING_DIR = os.path.join(BASE_DIR, "training")

# نفس المعمارية المستخدمة في التدريب على Kaggle
class HieroglyphEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_len=512, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        x = self.embed(x) * math.sqrt(self.embed.embedding_dim)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class HieroglyphTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_heads=8,
                 num_encoder_layers=4, num_decoder_layers=4,
                 ff_dim=512, max_len=512, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.src_embedding = HieroglyphEmbedding(vocab_size, embed_dim, max_len, dropout)
        self.tgt_embedding = HieroglyphEmbedding(vocab_size, embed_dim, max_len, dropout)
        self.transformer = nn.Transformer(
            d_model=embed_dim, nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=ff_dim, dropout=dropout, batch_first=True)
        self.output_layer = nn.Linear(embed_dim, vocab_size)
    def forward(self, src, tgt, src_padding_mask=None, tgt_padding_mask=None):
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(src.device)
        src_emb = self.src_embedding(src)
        tgt_emb = self.tgt_embedding(tgt)
        out = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask,
                               src_key_padding_mask=src_padding_mask,
                               tgt_key_padding_mask=tgt_padding_mask)
        return self.output_layer(out)

def load_tokenizer():
    with open(os.path.join(TOKENIZER_DIR, "tokenizer.json"), encoding="utf-8") as f:
        return json.load(f)

def translate(hieroglyphs, model, vocab, id_to_token, max_len=50):
    model.eval()
    bos = vocab["<BOS>"]
    eos = vocab["<EOS>"]
    unk = vocab["<UNK>"]
    src = [bos] + [vocab.get(c, unk) for c in hieroglyphs] + [eos]
    src_tensor = torch.tensor([src], dtype=torch.long)
    tgt = [bos]
    with torch.no_grad():
        for _ in range(max_len):
            tgt_tensor = torch.tensor([tgt], dtype=torch.long)
            output = model(src_tensor, tgt_tensor)
            logits = output[0, -1, :]
            for token_id in set(tgt):
                logits[token_id] -= 2.0
            next_token = logits.argmax().item()
            if next_token == eos:
                break
            tgt.append(next_token)
    return ''.join([id_to_token.get(str(i), '') for i in tgt[1:]])

def main():
    print("=== اختبار ترجمة الهيروغليفية ===\n")

    tokenizer = load_tokenizer()
    vocab = tokenizer["vocab"]
    id_to_token = tokenizer["id_to_token"]
    vocab_size = tokenizer["vocab_size"]

    model = HieroglyphTransformer(vocab_size=vocab_size)
    model_path = os.path.join(TRAINING_DIR, "best_model_v5.pt")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    print("✅ النموذج محمّل\n")

    test_cases = [
        "𓄿𓏤",
        "𓄿",
        "𓄿𓇋𓏲𓊡",
        "𓄿𓄿𓈇𓏤",
        "𓆑𓂤𓇛𓈙",
    ]

    print("الهيروغليفية  →  الترجمة")
    print("-" * 40)
    for hiero in test_cases:
        result = translate(hiero, model, vocab, id_to_token)
        print(f"{hiero}  →  {result}")

if __name__ == "__main__":
    main()
