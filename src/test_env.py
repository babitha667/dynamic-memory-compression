import torch
import os
from transformers import AutoTokenizer, AutoConfig

print("--- System Check ---")
print(f"PyTorch Version: {torch.__version__}")
print(f"PyTorch CPU-only Mode: {'Active' if not torch.cuda.is_available() else 'Inactive'}")

local_cache_dir = os.path.join(os.getcwd(), "data", "hf_cache")
model_ckpt = "gpt2"

print("\n--- Initializing HuggingFace Structure ---")
try:
    # We use a lightweight open-source model configuration to test the framework setup
    print(f"Loading tokenizer configuration for: {model_ckpt}...")
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt, cache_dir=local_cache_dir)
    
    test_text = "Testing memory compression workspace."
    tokens = tokenizer.encode(test_text)
    
    print("✓ Environment status: SUCCESS")
    print(f"Successfully processed test string into {len(tokens)} tokens!")
except Exception as e:
    print(f"✗ Setup verification failed: {e}")
