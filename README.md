# 🧠 Advanced Sentiment Analysis System

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-blue?style=flat-square&logo=python)](https://python.org)
[![Built with Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-red?style=flat-square&logo=streamlit)](https://streamlit.io)
[![Powered by Transformers](https://img.shields.io/badge/Powered%20by-🤗%20Transformers-yellow?style=flat-square)](https://huggingface.co/transformers)
[![AI Explanations](https://img.shields.io/badge/AI%20Explanations-Euron.one-green?style=flat-square&logo=robot)](https://euron.one)

> A production-ready sentiment analysis system built with fine-tuned BERT models, featuring real-time predictions and AI-powered explanations.

![Sentiment Analysis Demo](https://img.shields.io/badge/Status-Production%20Ready-brightgreen?style=for-the-badge)

## 🎯 **Perfect Dataset: Sentiment140**

**🔹 Overview:**
- **Name:** Sentiment140 (Stanford Research Dataset)
- **Size:** 1.6 million tweets
- **Labels:** Positive (4), Negative (0), Neutral (2, optional)
- **Format:** CSV — [sentiment, id, date, query, user, text]

**📚 Description:**
This dataset was created by Stanford researchers and is widely used to train and benchmark sentiment models.
- Trained on **real-world Twitter data**
- Captures the nuance of informal language, emojis, slang — perfect for robust models
- Works extremely well with transformer models due to its size and diversity

## 🚀 **Key Features**

- **🤖 Fine-tuned DistilBERT** for fast and accurate sentiment classification
- **📊 Interactive Web Interface** built with Streamlit
- **🔄 Batch Processing** for analyzing multiple texts at once
- **🧠 AI-Powered Explanations** using Euron.one API
- **☁️ Word Cloud Visualization** for sentiment analysis results
- **📁 Multi-format Support** (CSV, Excel, TXT files)
- **⚡ GPU Acceleration** with CUDA support
- **🛡️ Production-Ready** error handling and logging

## 🏗️ **System Architecture**

```
📁 Project Structure
├── 📄 app.py                    # Streamlit web interface
├── 📁 config/
│   └── config.yaml              # Configuration settings
├── 📁 data/
│   ├── raw/                     # Original dataset
│   ├── processed/               # Cleaned data
│   └── features/                # Tokenized features
├── 📁 models/
│   ├── checkpoints/             # Trained model weights
│   └── tokenizer/               # Model tokenizer
├── 📁 scripts/
│   ├── data_preprocessing.py    # Data cleaning pipeline
│   ├── feature_extraction.py    # Text tokenization
│   ├── train.py                 # Model training
│   ├── evaluate.py              # Model evaluation
│   └── llm_explanation.py       # AI explanations
├── 📁 utils/
│   └── config_loader.py         # Configuration management
└── 📁 logutils/
    └── logger.py                # Logging utilities
```

## ⚙️ **Training Configuration & Optimization**

### 🔧 **Hardware Constraints & Solutions**
- **GPU:** NVIDIA GeForce RTX 3060 (6GB VRAM)
- **Training Time:** ~8 hours for 2 epochs on 1.6M samples
- **Memory Optimization:** Gradient accumulation + mixed precision training

### 📊 **Training Parameters**
```yaml
Model: distilbert-base-uncased (66M parameters)
Epochs: 2 (limited by computational resources)
Batch Size: 8 (with gradient accumulation steps: 4)
Effective Batch Size: 32
Learning Rate: 2e-5
Max Sequence Length: 128
Mixed Precision: Enabled (AMP)
Device: CUDA (RTX 3060)
```

### 🎯 **Performance Optimizations**
- **Mixed Precision Training** (AMP) for 2x memory efficiency
- **Gradient Accumulation** to simulate larger batch sizes
- **DistilBERT** instead of BERT (60% smaller, 60% faster)
- **Dynamic Padding** to reduce computational overhead
- **Efficient Data Loading** with PyTorch DataLoader

### 💡 **Resource Scaling Recommendations**
> **Note:** This model was trained for only 2 epochs due to hardware limitations. 
> For better performance, consider:
> - **More Epochs:** 3-5 epochs typically yield better results
> - **Larger GPU:** RTX 4080/4090 or A100 for faster training
> - **Cloud Training:** Use Google Colab Pro, AWS, or Azure ML
> - **Model Variants:** Try RoBERTa or larger BERT models

## 🚀 **Complete Training & Deployment Pipeline**

### **📋 Prerequisites**
- Python 3.8+
- CUDA-compatible GPU (RTX 3060 or better recommended)
- 8GB+ RAM
- ~20GB free disk space for dataset and models

### **🔧 Environment Setup**
```bash
# Clone the repository
git clone <repository-url>
cd sentiment_analysis_advanced

# Create virtual environment
python -m venv llmapp_latest
source llmapp_latest/bin/activate  # Linux/Mac
# or llmapp_latest\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### **📁 Dataset Preparation**
```bash
# Create project structure
python setup_project.py

# Download Sentiment140 dataset (Manual step required)
# Visit: http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip
# Extract and place files:
# - training.1600000.processed.noemoticon.csv → data/raw/
# - testdata.manual.2009.06.14.csv → data/raw/

# Alternative: Use wget (Linux/Mac)
cd data/raw/
wget http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip
unzip trainingandtestdata.zip
```

### **⚙️ Environment Configuration**
Create `.env` file in project root:
```env
EURIAI_API_KEY=your_euron_api_key_here
ENV=development
HUGGINGFACE_TOKEN=your_hf_token_here
```

---

## 🔄 **Step-by-Step Training Pipeline**

### **Step 1: 🧹 Data Preprocessing**
```bash
python scripts/data_preprocessing.py
```
**What it does:**
- Loads raw Sentiment140 dataset (1.6M tweets)
- Maps sentiment labels: 0 (negative) → 0, 4 (positive) → 1
- Cleans text data and removes invalid entries
- Saves processed data to `data/processed/processed_data.csv`
- **Expected runtime:** 2-3 minutes

**✅ Success indicators:**
```
📊 Loaded dataset with 1600000 rows and 6 columns
🔁 Mapped sentiment labels: {0: 800000, 4: 800000} ➡️ {0: 800000, 1: 800000}
✅ Preprocessed data saved to data/processed/processed_data.csv
```

---

### **Step 2: 🔤 Feature Extraction**
```bash
python scripts/feature_extraction.py
```
**What it does:**
- Tokenizes text using DistilBERT tokenizer
- Converts text to numerical input IDs and attention masks
- Processes data in batches to handle memory efficiently
- Saves tokenized features to `data/features/encoded_features.pt`
- **Expected runtime:** 15-20 minutes

**✅ Success indicators:**
```
🧠 Loaded tokenizer: distilbert-base-uncased
📦 Starting tokenization in batches of 512...
🧪 Concatenating all batches...
✅ Encoded features saved to data/features/encoded_features.pt
```

---

### **Step 3: 🏗️ Tokenizer Setup (If Required)**

If you encounter "tokenizer not found" errors, run this script:

```python
# scripts/setup_tokenizer.py - Create this file
import os
from pathlib import Path
from transformers import AutoTokenizer
from utils.config_loader import load_config
from logutils.logger import get_logger

logger = get_logger("tokenizer_setup")

def setup_tokenizer():
    try:
        config = load_config()
        model_name = config["training"]["model_name"]
        tokenizer_dir = Path(config["paths"]["tokenizer_dir"])
        
        logger.info(f"🔧 Setting up tokenizer: {model_name}")
        
        # Create tokenizer directory
        os.makedirs(tokenizer_dir, exist_ok=True)
        
        # Download and save tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(tokenizer_dir)
        
        logger.info(f"✅ Tokenizer saved to: {tokenizer_dir}")
        
    except Exception as e:
        logger.exception(f"❌ Tokenizer setup failed: {e}")

if __name__ == "__main__":
    setup_tokenizer()
```

**Run tokenizer setup:**
```bash
python scripts/setup_tokenizer.py
```

---

### **Step 4: 🤖 Model Training**
```bash
python scripts/train.py
```
**What it does:**
- Loads tokenized features from Step 2
- Initializes DistilBERT model for sequence classification
- Trains for 2 epochs with mixed precision (AMP)
- Saves model checkpoints after each epoch
- **Expected runtime:** 6-8 hours on RTX 3060

**⚙️ Training Configuration:**
- **Model:** DistilBERT-base-uncased (66M parameters)
- **Batch Size:** 8 (effective 32 with gradient accumulation)
- **Learning Rate:** 2e-5
- **Epochs:** 2
- **Optimization:** AdamW + Linear warmup scheduler
- **Mixed Precision:** Enabled for memory efficiency

**✅ Success indicators:**
```
💻 Using device: cuda
🧠 Model: distilbert-base-uncased | Epochs: 2 | Batch size: 8
🔁 Starting Epoch 1/2
📉 Epoch 1 completed. Avg Loss: 0.421 | Time: 4.2 hours
💾 Checkpoint saved at: models/checkpoints/epoch_1
🔁 Starting Epoch 2/2
📉 Epoch 2 completed. Avg Loss: 0.285 | Time: 3.8 hours
💾 Checkpoint saved at: models/checkpoints/epoch_2
✅ Training completed successfully!
```

**📁 Output Structure:**
```
models/
├── checkpoints/
│   ├── epoch_1/          # First epoch checkpoint
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   └── tokenizer files...
│   └── epoch_2/          # Final model (used for inference)
│       ├── config.json
│       ├── pytorch_model.bin
│       └── tokenizer files...
```

---

### **Step 5: 📊 Model Evaluation**
```bash
python scripts/evaluate.py
```
**What it does:**
- Loads the trained model from `epoch_2`
- Evaluates on Stanford test dataset (359 manually annotated samples)
- Computes accuracy, precision, recall, and F1-score
- **Expected runtime:** 2-3 minutes

**✅ Expected Results:**
```
✅ Evaluation Results:
🎯 Accuracy: 0.7850
🎯 Precision: 0.7920
🎯 Recall: 0.7780
🎯 F1 Score: 0.7850
```

---

### **Step 6: 🚀 Launch Web Application**
```bash
streamlit run app.py
```
**What it does:**
- Loads the trained model from `models/checkpoints/epoch_2`
- Starts interactive web interface on `http://localhost:8501`
- Provides real-time sentiment analysis with AI explanations

**🌐 Web Interface Features:**
- **Single Text Analysis:** Real-time predictions with confidence scores
- **Batch Analysis:** Upload CSV/Excel/TXT files for bulk processing
- **AI Explanations:** Powered by Euriai API
- **Word Clouds:** Visual representation of sentiment patterns
- **Results Download:** Export predictions as CSV

---

## 🐛 **Troubleshooting Guide**

### **❌ Common Issues & Solutions**

#### **1. CUDA Out of Memory**
```bash
# Reduce batch size in config.yaml
batch_size: 4  # Instead of 8
gradient_accumulation_steps: 8  # Instead of 4
```

#### **2. Dataset Not Found**
```bash
# Verify file placement
ls data/raw/
# Should show: training.1600000.processed.noemoticon.csv
```

#### **3. Model Not Found During Inference**
```bash
# Check if training completed successfully
ls models/checkpoints/epoch_2/
# Should show: config.json, pytorch_model.bin, etc.
```

#### **4. Tokenizer Errors**
```bash
# Run tokenizer setup
python scripts/setup_tokenizer.py
```

#### **5. API Key Errors**
```bash
# Verify .env file exists and contains:
EURIAI_API_KEY=your_actual_api_key
```

---

## 📈 **Performance Monitoring**

### **Training Progress Tracking**
Monitor these metrics during training:
- **Loss Reduction:** Should decrease from ~0.69 to ~0.28
- **GPU Memory:** Should stay under 6GB on RTX 3060
- **Training Speed:** ~200-300 samples/second

### **Memory Usage Tips**
```bash
# Monitor GPU usage
nvidia-smi -l 1

# If memory issues occur:
# 1. Reduce batch_size to 4 or 2
# 2. Increase gradient_accumulation_steps proportionally
# 3. Enable gradient_checkpointing in train.py
```

---

## 🎯 **Quick Validation Pipeline**

After completing all steps, validate your setup:

```bash
# 1. Quick preprocessing test
python -c "from scripts.data_preprocessing import preprocess_data; preprocess_data()"

# 2. Quick feature extraction test (on small sample)
python -c "import torch; print('Features loaded:', torch.load('data/features/encoded_features.pt')['input_ids'].shape)"

# 3. Quick model test
python -c "from app import predict_sentiment; print(predict_sentiment('I love this!'))"

# 4. Launch app
streamlit run app.py
```

---

## 🚀 **Production Deployment**

### **Docker Deployment**
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### **Cloud Deployment Options**
- **Streamlit Cloud:** Direct GitHub integration
- **Heroku:** Web app deployment
- **AWS EC2:** Full control with GPU instances
- **Google Colab:** Free GPU training environment

## 🎨 **Web Interface Features**

### **📝 Single Text Analysis**
- Real-time sentiment prediction
- Confidence scores with progress bars
- AI-powered explanations via Euron.one API
- Support for texts up to 1000 characters

### **📄 Batch Analysis**
- Upload CSV, Excel, or TXT files
- Progress tracking for large datasets
- Results summary with sentiment distribution
- Word cloud visualization
- Downloadable results in CSV format

### **📊 Supported File Formats**
- **CSV:** Must contain a 'text' column
- **Excel (.xlsx/.xls):** Must contain a 'text' column
- **TXT:** One text per line (no headers needed)

## 🔬 **Model Performance**

### **Evaluation Metrics**
```
Dataset: Sentiment140 Test Set
Accuracy: 78.5%
Precision: 79.2% (Binary Classification)
Recall: 77.8%
F1-Score: 78.5%
```

### **Training Insights**
- **Epoch 1:** Training Loss: 0.421
- **Epoch 2:** Training Loss: 0.285
- **Model Size:** 268MB (compressed)
- **Inference Speed:** ~50ms per text on RTX 3060

## 🤖 **AI-Powered Explanations**

This system uses **Euron.one API** instead of OpenAI for generating explanations:

### **Why Euron.one?**
- ✅ **Cost-effective** alternative to OpenAI
- ✅ **Fast response times** for real-time explanations
- ✅ **Specialized models** for text analysis tasks
- ✅ **API compatibility** with OpenAI format

### **Example Explanation**
```
Input: "This movie was absolutely fantastic!"
Prediction: Positive 😊 (Confidence: 0.9234)

AI Explanation: "The text expresses strong positive sentiment through 
the use of emphatic language. The word 'absolutely' serves as an 
intensifier, while 'fantastic' is a highly positive adjective that 
clearly indicates enthusiasm and satisfaction."
```

## 📁 **Dataset Information**

### **Sentiment140 Dataset Details**
- **Source:** Stanford University
- **Original Paper:** "Twitter Sentiment Classification using Distant Supervision"
- **Download:** [CS Stanford](http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip)
- **License:** Research/Educational Use

### **Data Distribution**
- **Training Samples:** 1,600,000
- **Positive Samples:** 800,000 (50%)
- **Negative Samples:** 800,000 (50%)
- **Test Samples:** 359 (manually annotated)

## 🔧 **Configuration**

### **config/config.yaml**
```yaml
training:
  model_name: "distilbert-base-uncased"
  max_length: 128
  batch_size: 8
  gradient_accumulation_steps: 4
  learning_rate: 2e-5
  epochs: 2
  device: "cuda"

paths:
  raw_data: "data/raw/training.1600000.processed.noemoticon.csv"
  model_dir: "models/checkpoints"
  tokenizer_dir: "models/tokenizer"
```

## 🐛 **Troubleshooting**

### **Common Issues**
1. **CUDA Out of Memory**
   - Reduce batch_size to 4 or 2
   - Enable gradient_checkpointing
   - Use CPU training (slower)

2. **Model Not Found**
   - Run the training pipeline first
   - Check model paths in config.yaml

3. **API Key Errors**
   - Verify EURIAI_API_KEY in .env file
   - Check API quota limits

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Special Thanks**

<table>
<tr>
<td align="center">
<img src="https://github.com/sudhanshu-kumar.png" width="80px;" alt="API Provider"/><br />
<sub><b>⚡ Sudhanshu Kumar</b></sub><br />
<sub>Euriai API Provider</sub><br />
<a href="https://www.linkedin.com/in/-sudhanshu-kumar/?originalSubdomain=in">LinkedIn</a> • <a href="https://euron.one/">Euron.one</a>
</td>
<td align="center">
<img src="https://github.com/bappy-ahmed.png" width="80px;" alt="Mentor"/><br />
<sub><b>💡 Bappy Ahmed</b></sub><br />
<sub>Mentor & Inspiration</sub><br />
<a href="https://www.linkedin.com/in/boktiarahmed73/overlay/about-this-profile/">LinkedIn</a>
</td>
</tr>
</table>

### 🔗 **Powered By**

<div align="center">

[![Euriai API](https://img.shields.io/badge/Powered_by-Euriai_API-blue?style=for-the-badge&logo=lightning)](https://euron.one/)
[![Streamlit](https://img.shields.io/badge/Built_with-Streamlit-red?style=for-the-badge&logo=streamlit)](https://streamlit.io/)

</div>

### 🏆 **Acknowledgments**

- **Stanford University** for the Sentiment140 dataset
- **Hugging Face** for the Transformers library
- **Euriai/Euron.one** for AI explanation services
- **Streamlit** for the web interface framework

## 📈 **Future Improvements**

- [ ] **Multi-language support** (Spanish, French, German)
- [ ] **Real-time streaming** sentiment analysis
- [ ] **Model ensemble** for improved accuracy
- [ ] **Active learning** with user feedback
- [ ] **REST API** for programmatic access
- [ ] **Docker containerization** for easy deployment
- [ ] **Model quantization** for edge deployment

## 📞 **Contact**

**Rohit Bharti**
- GitHub: [@rohit-bharti]
- Email: rohit.bharti8211@gmail.com
- LinkedIn: [Connect with me]

---

*© 2025 Rohit Bharti. Built with ❤️ using Python, Transformers, and Streamlit.*

**Powered by:** 
- 🤗 **Hugging Face Transformers**
- 🚀 **Streamlit** 
- 🧠 **Euron.one AI**