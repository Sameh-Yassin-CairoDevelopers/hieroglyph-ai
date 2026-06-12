# transformer.py
# نموذج Transformer مخصص للهيروغليفية

import sys
import io
import torch
import torch.nn as nn
import math

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

class HieroglyphEmbedding(nn.Module):
    """طبقة التضمين مع Positional Encoding"""
    
    def __init__(self, vocab_size, embed_dim, max_len=512, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        
        # Positional Encoding
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = self.embed(x) * math.sqrt(self.embed.embedding_dim)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class HieroglyphTransformer(nn.Module):
    """
    نموذج Transformer لترجمة الهيروغليفية
    المدخل:  رموز هيروغليفية
    المخرج:  معنى بالعربي
    """
    
    def __init__(self, vocab_size, embed_dim=512, num_heads=8,
                 num_encoder_layers=6, num_decoder_layers=6,
                 ff_dim=1024, max_len=1024, dropout=0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        
        # طبقات التضمين
        self.src_embedding = HieroglyphEmbedding(vocab_size, embed_dim, max_len, dropout)
        self.tgt_embedding = HieroglyphEmbedding(vocab_size, embed_dim, max_len, dropout)
        
        # Transformer الأساسي
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        
        # طبقة الإخراج
        self.output_layer = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, src, tgt, src_padding_mask=None, tgt_padding_mask=None):
        # إنشاء Causal Mask للـ Decoder
        tgt_len = tgt.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len)
        
        # التضمين
        src_emb = self.src_embedding(src)
        tgt_emb = self.tgt_embedding(tgt)
        
        # Transformer
        out = self.transformer(
            src_emb, tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        
        return self.output_layer(out)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def test_model():
    """اختبار بسيط للتأكد أن النموذج يعمل"""
    print("=== اختبار النموذج ===\n")
    
    vocab_size = 827
    model = HieroglyphTransformer(vocab_size=vocab_size)
    
    params = model.count_parameters()
    print(f"حجم النموذج: {params:,} parameter")
    print(f"حجم بالـ MB: {params * 4 / 1024 / 1024:.1f} MB")
    
    # اختبار بمدخلات وهمية
    batch_size = 2
    src_len = 10
    tgt_len = 15
    
    src = torch.randint(1, vocab_size, (batch_size, src_len))
    tgt = torch.randint(1, vocab_size, (batch_size, tgt_len))
    
    model.eval()
    with torch.no_grad():
        output = model(src, tgt)
    
    print(f"\nشكل المدخل:  {src.shape}")
    print(f"شكل المخرج:  {output.shape}")
    print(f"\n✅ النموذج يعمل بشكل صحيح")


if __name__ == "__main__":
    test_model()
