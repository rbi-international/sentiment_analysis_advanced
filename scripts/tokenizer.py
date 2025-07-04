# scripts/setup_tokenizer.py
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import os
from transformers import AutoTokenizer
from utils.config_loader import load_config
from logutils.logger import get_logger

logger = get_logger("tokenizer_setup")

def setup_tokenizer():
    """
    Downloads and saves the tokenizer to the specified directory.
    This is useful when the tokenizer is not found during training or inference.
    """
    try:
        logger.info("🔧 Setting up tokenizer...")
        
        # Load configuration
        config = load_config()
        model_name = config["training"]["model_name"]
        tokenizer_dir = Path(config["paths"]["tokenizer_dir"])
        
        logger.info(f"📥 Downloading tokenizer: {model_name}")
        logger.info(f"📁 Target directory: {tokenizer_dir}")
        
        # Create tokenizer directory if it doesn't exist
        os.makedirs(tokenizer_dir, exist_ok=True)
        
        # Download tokenizer from Hugging Face
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info("✅ Tokenizer downloaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to download tokenizer: {e}")
            logger.info("💡 Make sure you have internet connection and the model name is correct")
            return False
        
        # Save tokenizer to local directory
        try:
            tokenizer.save_pretrained(tokenizer_dir)
            logger.info(f"💾 Tokenizer saved to: {tokenizer_dir}")
        except Exception as e:
            logger.error(f"❌ Failed to save tokenizer: {e}")
            return False
        
        # Verify tokenizer was saved correctly
        saved_files = list(tokenizer_dir.glob("*"))
        if saved_files:
            logger.info("✅ Tokenizer setup completed successfully!")
            logger.info(f"📋 Files saved: {[f.name for f in saved_files]}")
            return True
        else:
            logger.error("❌ No files were saved. Setup failed.")
            return False
            
    except Exception as e:
        logger.exception(f"🔥 Unexpected error during tokenizer setup: {e}")
        return False

def verify_tokenizer():
    """
    Verify that the tokenizer can be loaded successfully.
    """
    try:
        config = load_config()
        tokenizer_dir = Path(config["paths"]["tokenizer_dir"])
        
        if not tokenizer_dir.exists():
            logger.warning(f"⚠️ Tokenizer directory not found: {tokenizer_dir}")
            return False
        
        # Try to load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
        
        # Test tokenization
        test_text = "This is a test sentence."
        tokens = tokenizer(test_text, return_tensors="pt")
        
        logger.info("✅ Tokenizer verification successful!")
        logger.info(f"🧪 Test tokenization result: {tokens['input_ids'].shape}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Tokenizer verification failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting tokenizer setup...")
    
    # Setup tokenizer
    success = setup_tokenizer()
    
    if success:
        print("\n🔍 Verifying tokenizer...")
        verify_success = verify_tokenizer()
        
        if verify_success:
            print("\n🎉 Tokenizer setup and verification completed successfully!")
            print("💡 You can now proceed with feature extraction and training.")
        else:
            print("\n❌ Tokenizer verification failed. Please check the logs.")
    else:
        print("\n❌ Tokenizer setup failed. Please check the logs and try again.")
        print("💡 Common solutions:")
        print("   - Check your internet connection")
        print("   - Verify the model name in config.yaml")
        print("   - Ensure you have sufficient disk space")