# app.py
# واجهة المستخدم لترجمة الهيروغليفية

import json
import os
import torch
import gradio as gr

BASE_DIR = os.path.dirname(__file__)
TOKENIZER_DIR = os.path.join(BASE_DIR, "tokenizer")
TRAINING_DIR  = os.path.join(BASE_DIR, "training")

import sys
sys.path.append(BASE_DIR)
from model.transformer import HieroglyphTransformer

# تحميل النموذج عند البداية
print("جاري تحميل النموذج...")

with open(os.path.join(TOKENIZER_DIR, "tokenizer.json"), encoding="utf-8") as f:
    tokenizer = json.load(f)

vocab        = tokenizer["vocab"]
id_to_token  = tokenizer["id_to_token"]
vocab_size   = tokenizer["vocab_size"]

model = HieroglyphTransformer(vocab_size=vocab_size)
model.load_state_dict(torch.load(
    os.path.join(TRAINING_DIR, "best_model.pt"),
    map_location="cpu"
))
model.eval()
print("✅ النموذج جاهز")


def translate(hieroglyphs: str) -> str:
    """ترجمة الرموز الهيروغليفية إلى العربية"""
    if not hieroglyphs.strip():
        return "أدخل رموزاً هيروغليفية"

    bos = vocab["<BOS>"]
    eos = vocab["<EOS>"]
    unk = vocab["<UNK>"]

    src = [bos] + [vocab.get(c, unk) for c in hieroglyphs.strip()] + [eos]
    src_tensor = torch.tensor([src], dtype=torch.long)

    tgt = [bos]
    with torch.no_grad():
        for _ in range(80):
            tgt_tensor = torch.tensor([tgt], dtype=torch.long)
            output     = model(src_tensor, tgt_tensor)
            logits     = output[0, -1, :]

            # منع التكرار
            for token_id in set(tgt):
                logits[token_id] -= 2.0

            next_token = logits.argmax().item()
            if next_token == eos:
                break
            tgt.append(next_token)

    result = ''.join([id_to_token.get(str(i), '') for i in tgt[1:]])
    return result if result else "لم أتمكن من الترجمة"


def search_dictionary(query: str) -> str:
    """البحث في القاموس المحلي"""
    if not query.strip():
        return "أدخل كلمة للبحث"

    results = []
    data_path = os.path.join(BASE_DIR, "data", "processed", "dictionary_clean.json")

    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)

    query = query.strip().lower()
    for word in data:
        meaning_en = word.get("meaning_en", "").lower()
        meaning_ar = word.get("meaning_ar", "")
        trans      = word.get("transliteration", "")
        hiero      = word.get("hieroglyphs", "")

        if query in meaning_en or query in meaning_ar or query in trans:
            results.append(
                f"🔤 {hiero}  |  {trans}  |  {meaning_ar}  |  {meaning_en}"
            )

        if len(results) >= 10:
            break

    if results:
        return "\n".join(results)
    return "لم يُعثر على نتائج"


# بناء الواجهة
with gr.Blocks(title="Hieroglyph AI") as app:

    gr.Markdown("""
    # 𓂀 Hieroglyph AI
    ### أول نظام ذكاء اصطناعي عربي لقراءة وترجمة النقوش الفرعونية
    """)

    with gr.Tabs():

        # تبويب الترجمة
        with gr.Tab("🔤 ترجمة الهيروغليفية"):
            gr.Markdown("أدخل الرموز الهيروغليفية وسيترجمها النموذج للعربية")

            with gr.Row():
                input_box = gr.Textbox(
                    label="الرموز الهيروغليفية",
                    placeholder="𓄿𓏤",
                    lines=2
                )
                output_box = gr.Textbox(
                    label="الترجمة العربية",
                    lines=2,
                    interactive=False
                )

            translate_btn = gr.Button("ترجم", variant="primary")

            gr.Examples(
                examples=[["𓄿𓏤"], ["𓄿𓇋𓏲𓊡"], ["𓄿𓄿𓈇𓏤"]],
                inputs=input_box
            )

            translate_btn.click(
                fn=translate,
                inputs=input_box,
                outputs=output_box
            )

        # تبويب البحث
        with gr.Tab("🔍 البحث في القاموس"):
            gr.Markdown("ابحث بالعربية أو الإنجليزية أو النطق")

            search_box    = gr.Textbox(label="كلمة البحث", placeholder="vulture أو نسر")
            search_btn    = gr.Button("بحث", variant="primary")
            search_output = gr.Textbox(label="النتائج", lines=10, interactive=False)

            search_btn.click(
                fn=search_dictionary,
                inputs=search_box,
                outputs=search_output
            )

        # تبويب عن المشروع
        with gr.Tab("ℹ️ عن المشروع"):
            gr.Markdown("""
            ## Hieroglyph AI

            مشروع مفتوح المصدر لبناء أول نموذج ذكاء اصطناعي عربي
            متخصص في قراءة وترجمة النقوش الفرعونية.

            ### البيانات
            - **6,541 كلمة** من مكتبة الإسكندرية
            - **689 رمز هيروغليفي** فريد
            - معاني بالعربية والإنجليزية

            ### النموذج
            - **5.9 مليون parameter**
            - مدرّب على CPU عادي
            - Transformer مخصص للهيروغليفية

            ### المطور
            Sameh Yassin — CairoDevelopers
            """)

app.launch(theme=gr.themes.Soft())
