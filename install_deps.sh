#!/bin/bash
set -e
cd /mnt/c/Users/majip/Downloads/xllm/legal_scanner_llm
source venv/bin/activate
echo "Upgrading pip..."
pip install --upgrade pip
echo "Installing Unsloth..."
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
echo "Installing xformers, trl, peft, accelerate, bitsandbytes..."
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
echo "Installing requirements.txt..."
pip install -r requirements.txt
echo "All dependencies installed successfully!"
