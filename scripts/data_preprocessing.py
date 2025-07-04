# scripts/data_preprocessing.py

import sys
from pathlib import Path
import sys
from pathlib import Path
import os
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from logutils.logger import get_logger
from utils.config_loader import load_config

logger = get_logger("data_preprocessing")

def preprocess_data():
    try:
        logger.info("🚀 Starting sentiment data preprocessing...")

        config = load_config()
        project_root = Path(__file__).resolve().parents[1]
        raw_data_path = os.path.join(project_root, config["paths"]["raw_data"])
        processed_data_path = os.path.join(project_root, config["paths"]["processed_data"])
        logger.debug(f"📂 Absolute raw data path resolved to: {raw_data_path}")



        logger.debug(f"📁 Raw data path: {raw_data_path}")
        logger.debug(f"📁 Processed data output path: {processed_data_path}")

        if not os.path.exists(raw_data_path):
            logger.error(f"❌ Raw data file not found at {raw_data_path}")
            return

        df = pd.read_csv(
            raw_data_path,
            encoding="latin-1",
            header=None,
            names=["sentiment", "id", "date", "query", "user", "text"]
        )
        logger.info(f"📊 Loaded dataset with {len(df)} rows and {df.shape[1]} columns")

        df = df[["sentiment", "text"]]
        logger.info("🧹 Dropped unused columns. Remaining: sentiment, text")

        original_counts = df["sentiment"].value_counts().to_dict()
        df["sentiment"] = df["sentiment"].map({0: 0, 4: 1})
        df = df.dropna(subset=["sentiment"])
        logger.info(f"🔁 Mapped sentiment labels: {original_counts} ➡️ {df['sentiment'].value_counts().to_dict()}")

        df["text"] = df["text"].astype(str).str.strip()
        logger.info("🧼 Cleaned text fields (stripped whitespace)")

        os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
        df.to_csv(processed_data_path, index=False)
        logger.success = getattr(logger, 'success', logger.info)
        logger.success(f"✅ Preprocessed data saved to {processed_data_path}")

    except Exception as e:
        logger.exception(f"🔥 Exception occurred during preprocessing: {e}")

if __name__ == "__main__":
    preprocess_data()
