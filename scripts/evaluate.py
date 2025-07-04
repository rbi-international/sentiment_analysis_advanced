import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from utils.config_loader import load_config
from logutils.logger import get_logger

logger = get_logger("evaluator")


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx]),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx]),
            "labels": torch.tensor(self.labels[idx])
        }


def evaluate():
    try:
        logger.info("🔍 Starting evaluation...")
        config = load_config()
        ROOT = Path(__file__).resolve().parent.parent
        model_dir = (ROOT / config["paths"]["model_dir"] / "epoch_2").resolve().as_posix()

        
        test_data_path = (ROOT / config["paths"]["test_data"]).resolve()

        if not test_data_path.exists():
            logger.error(f"❌ Test data not found at: {test_data_path}")
            return
        logger.info(f"✅ Test data found at: {test_data_path}")

        df = pd.read_csv(test_data_path, encoding='ISO-8859-1', header=None)
        df.columns = ["target", "ids", "date", "flag", "user", "text"]
        df = df[["target", "text"]].dropna()

        texts = df["text"].tolist()
        labels = [0 if label == 0 else 1 for label in df["target"]]  # 0 = negative, 4 = positive → 0 or 1

        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        dataset = SentimentDataset(texts, labels, tokenizer, config["training"]["max_length"])
        dataloader = DataLoader(dataset, batch_size=config["training"]["batch_size"])

        preds, true_labels = [], []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)

                preds.extend(predictions.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(true_labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, preds, average='binary')

        logger.info(f"✅ Evaluation Results:\n"
                    f"🎯 Accuracy: {acc:.4f}\n"
                    f"🎯 Precision: {precision:.4f}\n"
                    f"🎯 Recall: {recall:.4f}\n"
                    f"🎯 F1 Score: {f1:.4f}")

    except Exception as e:
        logger.exception(f"❌ Evaluation failed: {e}")


if __name__ == "__main__":
    evaluate()
