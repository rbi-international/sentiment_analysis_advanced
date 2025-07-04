import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_scheduler
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import time

from logutils.logger import get_logger
from utils.config_loader import load_config

logger = get_logger("model_trainer")

class SentimentDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.encodings["labels"][idx],
        }

def train():
    try:
        logger.info("🚀 Starting model training...")

        config = load_config()
        features_path = ROOT / config["paths"]["features_data"]
        model_dir = ROOT / config["paths"]["model_dir"]
        tokenizer_dir = ROOT / config["paths"]["tokenizer_dir"]
        model_name = config["training"]["model_name"]
        batch_size = int(config["training"]["batch_size"])
        lr = float(config["training"]["learning_rate"])
        epochs = int(config["training"]["epochs"])
        device = config["training"]["device"]

        device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
        logger.info(f"💻 Using device: {device}")

        if not os.path.exists(features_path):
            logger.error(f"❌ Encoded features file not found at: {features_path}")
            return

        encodings = torch.load(features_path)
        dataset = SentimentDataset(encodings)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        optimizer = AdamW(model.parameters(), lr=lr)
        scaler = GradScaler(device="cuda")  # ✅ Fixed

        logger.info(f"🧠 Model: {model_name} | Epochs: {epochs} | Batch size: {batch_size}")

        model.train()
        for epoch in range(epochs):
            logger.info(f"🔁 Starting Epoch {epoch+1}/{epochs}")
            total_loss = 0.0
            start_time = time.time()

            pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"🚀 Epoch {epoch + 1}", dynamic_ncols=True)
            for step, batch in pbar:
                batch = {k: v.to(device) for k, v in batch.items()}

                with autocast(device_type="cuda"):
                    outputs = model(**batch)
                    loss = outputs.loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})


            avg_loss = total_loss / len(dataloader)
            duration = time.time() - start_time
            logger.info(f"📉 Epoch {epoch+1} completed. Avg Loss: {avg_loss:.4f} | Time: {duration/60:.2f} min")

            # Save model checkpoint
            epoch_dir = os.path.join(model_dir, f"epoch_{epoch+1}")
            os.makedirs(epoch_dir, exist_ok=True)
            model.save_pretrained(epoch_dir)
            tokenizer.save_pretrained(epoch_dir)
            logger.info(f"💾 Checkpoint saved at: {epoch_dir}")

        logger.info("✅ Training completed successfully!")

    except Exception as e:
        logger.exception(f"🔥 Training failed due to: {e}")

if __name__ == "__main__":
    train()
