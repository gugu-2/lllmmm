#!/bin/bash
set -e

PROJECT_DIR="/mnt/c/Users/majip/Downloads/xllm/legal_scanner_llm"
cd "$PROJECT_DIR"

source venv_wsl/bin/activate

echo "============================================"
echo "  Installing remaining dependencies (fixed)"
echo "============================================"

# Pillow 12.2.0 is already installed from Unsloth.
# Install requirements individually, skipping Pillow and already-installed packages.
echo "[1/3] Installing app dependencies..."
pip install streamlit pandas pdfplumber docx2txt pytesseract fastapi uvicorn python-multipart

echo "[2/3] Skipping xformers (optional, not needed for modern Unsloth)..."

# Verify key imports
echo "[3/3] Verifying installation..."
python3 -c "
import torch
import unsloth
import transformers
import trl
import peft
import datasets
import accelerate
import bitsandbytes
import streamlit
import PIL
print('=== Installed Versions ===')
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('Unsloth:', unsloth.__version__)
print('Transformers:', transformers.__version__)
print('TRL:', trl.__version__)
print('PEFT:', peft.__version__)
print('Streamlit:', streamlit.__version__)
print('Pillow:', PIL.__version__)
print()
print('ALL IMPORTS SUCCESSFUL!')
"

echo ""
echo "============================================"
echo "  SETUP COMPLETE!"
echo "============================================"
