# Setup Guide: Legal "Red Flag" Scanner

This guide walks you through setting up the entire Legal Scanner LLM environment on a Windows machine with an NVIDIA GPU (e.g., RTX 5050).

## Prerequisites

1. **Windows 10/11**
2. **NVIDIA GPU** with at least 6GB VRAM.
3. **WSL 2 (Windows Subsystem for Linux)**: Unsloth requires Linux to compile its highly optimized Triton kernels.

---

## Step 1: Install Tesseract OCR (For Image Uploads)

Because we added the ability to scan images (JPG, PNG), your computer needs OCR software.
1. Download the Windows installer for Tesseract OCR from [UB-Mannheim's GitHub](https://github.com/UB-Mannheim/tesseract/wiki).
2. Install it. (Usually installs to `C:\Program Files\Tesseract-OCR`).
3. **Crucial:** Add `C:\Program Files\Tesseract-OCR` to your Windows System `PATH` environment variable so Python can find it.

---

## Step 2: Install WSL (Windows Subsystem for Linux)

If you haven't used WSL before, open **PowerShell as Administrator** and run:

```powershell
wsl --install
```

*Note: You may need to restart your computer after this finishes.*

Once installed, launch **Ubuntu** from your Windows Start Menu. It will ask you to create a Unix username and password.

---

## Step 3: Set Up the Project Environment (In WSL)

Navigate to the project folder where the files are stored. WSL mounts your `C:\` drive under `/mnt/c/`.

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3 python3-pip python3-venv git -y
cd /mnt/c/Users/majip/Downloads/xllm/legal_scanner_llm
python3 -m venv venv
source venv/bin/activate
```

---

## Step 4: Install Python Libraries

You must install Unsloth directly from their GitHub repository to get the latest features required for Gemma 2.

```bash
# Install Unsloth and specific dependencies for fast training
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes

# Install the app dependencies (including PDF and Image readers)
pip install -r requirements.txt
```

---

## Step 5: Training the Model

With the environment activated, you can now start the fine-tuning process. This will use the `legal_scanner_data.jsonl` file we prepared earlier.

```bash
python train_gemma.py
```

* **What to expect:** This process will take several hours depending on your GPU.
* **Output:** Once finished, it will create a folder called `lora_model`.

---

## Step 6: Running the Web Interface

Once training is complete and the `lora_model` folder exists, you can run the Streamlit app. You can do this from WSL or native Windows (if you installed the requirements on Windows).

```bash
streamlit run app.py
```

Open the URL provided in your terminal (usually `http://localhost:8501`) in your web browser. You can now drag and drop PDFs, Word Docs, or Images into the app!
