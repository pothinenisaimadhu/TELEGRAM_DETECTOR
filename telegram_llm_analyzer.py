# telegram_llm_analyzer.py
import io
from collections import defaultdict
from pymongo import MongoClient
from gridfs import GridFS
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoImageProcessor
import torch
import tempfile
import os
import whisper
from PIL import Image

# ===========================
# 📦 MongoDB Setup
# ===========================
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "telegram_data"

mongo_client = MongoClient(MONGO_URI)
db = mongo_client[DB_NAME]
messages_col = db["messages"]
fs = GridFS(db)

# ===========================
# 🤗 HuggingFace Pipelines
# ===========================
# Performance settings
USE_CUDA = torch.cuda.is_available()
DEVICE_ID = 0 if USE_CUDA else -1
torch.backends.cudnn.benchmark = True

# Summarizer (generic, small model) on device
summarizer = pipeline("summarization", model="t5-small", framework="pt", device=DEVICE_ID)

# ✅ Load your fine-tuned classifier instead of zero-shot
# ✅ Load your fine-tuned classifier instead of zero-shot
model_path = "./model_output"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# If tokenizer has no real pad token or pad==eos, add a distinct pad token to ensure reliable attention masks
if getattr(tokenizer, "pad_token", None) is None or getattr(tokenizer, "pad_token_id", None) == getattr(tokenizer, "eos_token_id", None):
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    try:
        model.resize_token_embeddings(len(tokenizer))
    except Exception:
        pass

# Move model to GPU and use fp16 where possible for faster inference
if USE_CUDA:
    try:
        model.to("cuda")
        model.half()
    except Exception:
        pass

# Keep a fallback pipeline but run it on the chosen device
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, framework="pt", device=DEVICE_ID)

# Determine tokenizer/model max length for safe truncation
try:
    MODEL_MAX_LENGTH = tokenizer.model_max_length
except Exception:
    MODEL_MAX_LENGTH = 512

def classify_text(text: str):
    """Tokenize explicitly (ensures attention_mask) and run model on correct device.

    Returns: dict with 'label' and 'score'.
    """
    # Ensure we don't feed arbitrarily long sequences
    enc = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=MODEL_MAX_LENGTH,
        return_tensors="pt",
    )

    device = next(model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}

    model.eval()
    with torch.no_grad():
        out = model(**enc)
        logits = out.logits
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    top_idx = int(probs.argmax())
    # map id -> label if available
    label = None
    if hasattr(model.config, "id2label"):
        label = model.config.id2label.get(top_idx, str(top_idx))
    else:
        label = str(top_idx)

    score = float(probs[top_idx])
    return {"label": label, "score": score}

# Image captioning + Whisper
try:
    image_processor = AutoImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning", use_fast=True)
except Exception:
    image_processor = None

# Load tokenizer for the image-captioning model and ensure it has a distinct PAD token
try:
    img_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    if getattr(img_tokenizer, "pad_token", None) is None or getattr(img_tokenizer, "pad_token_id", None) == getattr(img_tokenizer, "eos_token_id", None):
        img_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
except Exception:
    img_tokenizer = None

image_captioner = pipeline(
    "image-to-text",
    model="nlpconnect/vit-gpt2-image-captioning",
    feature_extractor=image_processor if image_processor is not None else None,
    tokenizer=img_tokenizer if img_tokenizer is not None else None,
    device=DEVICE_ID,
)

# Move image-caption model to GPU and use fp16 where possible
if USE_CUDA:
    try:
        image_captioner.model.to("cuda")
        image_captioner.model.half()
    except Exception:
        pass

whisper_model = whisper.load_model("base")  # Speech-to-text

# ===========================
# 📌 Media Handlers
# ===========================
def process_photo(file_id):
    try:
        file_data = fs.find_one({"filename": file_id})
        if not file_data:
            return None
        img = Image.open(io.BytesIO(file_data.read())).convert("RGB")
        # Resize to model expected input for faster processing while keeping aspect ratio
        try:
            img = img.resize((224, 224))
        except Exception:
            pass

        caption = image_captioner(img)[0]["generated_text"]
        return caption
    except Exception:
        return None

def process_audio_video(file_id):
    try:
        file_data = fs.find_one({"filename": file_id})
        if not file_data:
            return None
        # Use a temporary file compatible with Windows
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tf:
            tf.write(file_data.read())
            tmp_file = tf.name

        try:
            result = whisper_model.transcribe(tmp_file)
            return result.get("text", "").strip()
        finally:
            try:
                os.remove(tmp_file)
            except Exception:
                pass
    except Exception:
        return None

# ===========================
# 🧾 Chat-level Analyzer
# ===========================
def analyze_chats(limit=200):
    chats = defaultdict(list)

    # Collect messages grouped by chat
    for msg in messages_col.find().sort("date", -1).limit(limit):
        text = msg.get("text") or ""

        # Add media text if exists
        if msg.get("media_type") == "photo":
            caption = process_photo(msg["file_id"])
            if caption:
                text += f"\n[Image Caption] {caption}"

        elif msg.get("media_type") == "document":
            mime = msg.get("mime_type", "")
            if "video" in mime or "audio" in mime:
                transcript = process_audio_video(msg["file_id"])
                if transcript:
                    text += f"\n[Transcript] {transcript}"

        if text.strip():
            chats[msg.get("chat_name", "Unknown")].append(text)

    # Analyze each chat as a whole
    flagged = []
    for chat_name, texts in chats.items():
        combined_text = " ".join(texts)

        # Summarize chat content — use summarizer's tokenizer max length and explicit truncation
        try:
            # Use explicit tokenization + generation to ensure attention_mask is passed
            def summarize_text(text: str, max_length_out: int = 80, min_length_out: int = 25):
                tok = summarizer.tokenizer
                model_s = summarizer.model
                device = next(model_s.parameters()).device

                max_input = getattr(tok, "model_max_length", 1024)
                enc = tok(
                    text,
                    truncation=True,
                    max_length=max_input,
                    return_tensors="pt",
                )
                enc = {k: v.to(device) for k, v in enc.items()}

                gen = model_s.generate(
                    **enc,
                    max_length=max_length_out,
                    min_length=min_length_out,
                    do_sample=False,
                )
                out = tok.decode(gen[0], skip_special_tokens=True)
                return out

            sum_max = min(80, getattr(summarizer.tokenizer, "model_max_length", 80))
            summary = summarize_text(combined_text, max_length_out=sum_max, min_length_out=25)
        except Exception:
            summary = combined_text[:500]

        # Use explicit tokenization + model call to ensure attention_mask is passed and inputs are truncated
        try:
            classification = classify_text(combined_text)
            top_label = classification["label"]
            score = classification["score"]
        except Exception:
            # fallback to pipeline (still passes truncation/padding)
            classification = classifier(combined_text, truncation=True, padding=True, max_length=MODEL_MAX_LENGTH)[0]
            top_label = classification["label"]
            score = classification["score"]

        flagged.append({
            "chat_name": chat_name,
            "messages_count": len(texts),
            "summary": summary,
            "classification": top_label,
            "confidence": round(score, 3)
        })

    return flagged

# ===========================
# 🚀 Runner
# ===========================
if __name__ == "__main__":
    results = analyze_chats(limit=500)

    print("\n📌 Chat-level Analysis:\n")
    for r in results:
        print(f"💬 Chat: {r['chat_name']} ({r['messages_count']} msgs)")
        print(f"  📝 Summary: {r['summary']}")
        print(f"  🚩 Classified as: {r['classification']} ({r['confidence']})")
        print("-" * 60)
