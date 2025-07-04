# app.py
import streamlit as st
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils.config_loader import load_config
from logutils.logger import get_logger
from scripts.llm_explanation import get_sentiment_explanation
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Setup logger
logger = get_logger()

# Load config and model
config = load_config()
model_dir = Path(config["paths"]["model_dir"]) / "epoch_2"
tokenizer_dir = Path(config["paths"]["tokenizer_dir"])
device = torch.device(config["training"]["device"])

try:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    logger.info("✅ Model and tokenizer loaded successfully.")
except Exception as e:
    logger.exception(f"❌ Failed to load model/tokenizer: {e}")
    st.error("Model loading failed. Check logs.")
    st.stop()

# Inference
@st.cache_data(show_spinner=False)
def predict_sentiment(text):
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()

        confidence = probs[0][predicted_class].item()
        if abs(probs[0][1] - 0.5) < 0.1:
            label = "Neutral 😐"
        elif predicted_class == 1:
            label = "Positive 😊"
        else:
            label = "Negative 😞"

        return label, confidence, probs
    except Exception as e:
        logger.exception(f"❌ Prediction failed: {e}")
        return "Prediction error", 0.0, None

# Word Cloud Generator
def generate_wordcloud(df):
    sentiments = df["prediction"].unique()
    for sentiment in sentiments:
        text = " ".join(df[df["prediction"] == sentiment]["text"])
        wc = WordCloud(width=800, height=400, background_color="white").generate(text)
        st.subheader(f"☁️ Word Cloud - {sentiment}")
        fig, ax = plt.subplots()
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

# UI
st.set_page_config(page_title="Sentiment Analysis App", layout="centered")
st.title("🧠 Sentiment Analysis App")
st.markdown("Enter a sentence to predict its sentiment using your fine-tuned model and get an explanation.")

# Single text input
txt = st.text_area("💬 Enter your text here:", height=150)

# File upload
uploaded_file = st.file_uploader("📄 Or upload a CSV/XLSX file for batch prediction:", type=["csv", "xlsx"])

if st.button("🚀 Predict"):
    if txt.strip():
        label, confidence, probs = predict_sentiment(txt)
        st.success(f"**Prediction:** {label}")
        st.info(f"**Confidence:** {confidence:.4f}")

        with st.spinner("🧠 Generating explanation..."):
            explanation = get_sentiment_explanation(txt, label, confidence)
            st.markdown("### 📘 Explanation")
            st.write(explanation)

    elif uploaded_file:
        with st.spinner("📊 Processing uploaded file..."):
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file format.")
                st.stop()

            if "text" not in df.columns:
                st.error("Uploaded file must have a 'text' column.")
                st.stop()

            results = []
            for t in df["text"]:
                label, conf, _ = predict_sentiment(t)
                results.append((t, label, conf))

            pred_df = pd.DataFrame(results, columns=["text", "prediction", "confidence"])
            st.dataframe(pred_df.head(20))
            generate_wordcloud(pred_df)
    else:
        st.warning("Please enter text or upload a file.")
