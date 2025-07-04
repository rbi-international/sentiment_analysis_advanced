import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

from logutils.logger import get_logger
from utils.config_loader import load_config

logger = get_logger("feature_extraction")

BATCH_SIZE = 512  # You can reduce this to 256 if needed

def extract_features():
    try:
        logger.info("🔍 Starting batch-wise feature extraction using Hugging Face tokenizer")

        config = load_config()
        project_root = Path(__file__).resolve().parents[1]
        processed_path = os.path.join(project_root, config["paths"]["processed_data"])
        features_path = os.path.join(project_root, config["paths"]["features_data"])
        model_name = config["training"]["model_name"]
        max_length = config["training"]["max_length"]

        if not os.path.exists(processed_path):
            logger.error(f"❌ Processed data file not found at: {processed_path}")
            return

        df = pd.read_csv(processed_path)
        logger.info(f"📥 Loaded {len(df)} rows from processed data")

        texts = [str(t) for t in df["text"].tolist()]
        labels = df["sentiment"].tolist()

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info(f"🧠 Loaded tokenizer: {model_name}")

        all_encodings = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }

        logger.info(f"📦 Starting tokenization in batches of {BATCH_SIZE}...")
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="🧠 Tokenizing"):
            batch_texts = texts[i:i + BATCH_SIZE]
            batch_labels = labels[i:i + BATCH_SIZE]

            enc = tokenizer(
                batch_texts,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )

            all_encodings["input_ids"].append(enc["input_ids"])
            all_encodings["attention_mask"].append(enc["attention_mask"])
            all_encodings["labels"].append(torch.tensor(batch_labels))

        logger.info("🧪 Concatenating all batches...")
        final_encodings = {
            "input_ids": torch.cat(all_encodings["input_ids"]),
            "attention_mask": torch.cat(all_encodings["attention_mask"]),
            "labels": torch.cat(all_encodings["labels"])
        }

        os.makedirs(os.path.dirname(features_path), exist_ok=True)
        torch.save(final_encodings, features_path)
        logger.success = getattr(logger, 'success', logger.info)
        logger.success(f"✅ Encoded features saved to {features_path}")

    except Exception as e:
        logger.exception(f"🔥 Exception during batch feature extraction: {e}")

if __name__ == "__main__":
    extract_features()
