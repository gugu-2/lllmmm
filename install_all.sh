#!/bin/bash
set -e

PROJECT_DIR="/mnt/c/Users/majip/Downloads/xllm/legal_scanner_llm"
cd "$PROJECT_DIR"

echo "============================================"
echo "  Legal Scanner LLM - Full Dependency Setup"
echo "============================================"

# Activate the virtual environment
source venv_wsl/bin/activate
echo "[1/5] Virtual environment activated."

# Upgrade pip
echo "[2/5] Upgrading pip..."
pip install --upgrade pip

# Install Unsloth from GitHub
echo "[3/5] Installing Unsloth (this may take a while)..."
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Install training stack (pinned versions, no-deps to avoid conflicts)
echo "[4/5] Installing xformers, trl, peft, accelerate, bitsandbytes..."
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes

# Install app requirements
echo "[5/5] Installing requirements.txt..."
pip install -r requirements.txt

echo ""
echo "============================================"
echo "  ALL DEPENDENCIES INSTALLED SUCCESSFULLY!"
echo "============================================"
