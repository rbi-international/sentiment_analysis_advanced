import os

folders = [
    "config",
    "data/raw",
    "data/processed",
    "data/features",
    "models/checkpoints",
    "models/tokenizer",
    "scripts",
    "notebooks",
    "utils",
    "logs"
]

files = {
    ".env": """# Environment variables
ENV="development"
HUGGINGFACE_API_KEY=""
""",
    "config/config.yaml": """paths:
  raw_data: "data/raw/training.1600000.processed.noemoticon.csv"
  processed_data: "data/processed/processed_data.csv"
  features_data: "data/features/encoded_features.pt"
  model_dir: "models/checkpoints"
  tokenizer_dir: "models/tokenizer"

training:
  model_name: "bert-base-uncased"
  max_length: 128
  batch_size: 32
  learning_rate: 2e-5
  epochs: 5
  device: "cuda"

logging:
  log_file: "logs/pipeline.log"
  log_level: "INFO"
""",
    "requirements.txt": """torch
transformers
pandas
numpy
scikit-learn
tqdm
streamlit
PyYAML
""",
    "README.md": "# Sentiment Analysis System\nThis is a modular, production-grade sentiment analysis system built using fine-tuned BERT/RoBERTa models."
}

def create_structure():
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"📁 Created folder: {folder}")
        
    for file_path, content in files.items():
        with open(file_path, "w") as f:
            f.write(content)
            print(f"📄 Created file: {file_path}")

if __name__ == "__main__":
    create_structure()
