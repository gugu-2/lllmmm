# Legal Red Flag Scanner

An end-to-end local LLM application that scans legal contracts for specific clauses and risks.

## Phase 1: Data Preparation
You have already completed this! The `legal_scanner_data.jsonl` file contains thousands of lawyer-annotated examples.

## Phase 2: Training (Requires WSL / Linux)

Unsloth (the library we use to fit Gemma 2 9B onto an RTX 5050) requires a Linux environment. If you are on Windows, you must use **WSL (Windows Subsystem for Linux)**.

1. **Install WSL** (Open PowerShell as Administrator):
   ```bash
   wsl --install
   ```
   *Restart your PC if required.*

2. **Open Ubuntu (WSL)** from your start menu.

3. **Navigate to this project folder**:
   ```bash
   cd /mnt/c/Users/majip/Downloads/xllm/legal_scanner_llm
   ```

4. **Install the dependencies in WSL**:
   First, ensure you have Python and pip installed in WSL.
   ```bash
   sudo apt update
   sudo apt install python3-pip python3-venv
   python3 -m venv venv
   source venv/bin/activate
   ```
   
   Install Unsloth and standard requirements:
   ```bash
   pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
   pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
   pip install -r requirements.txt
   ```

5. **Start Training**:
   ```bash
   python train_gemma.py
   ```
   This will run for several hours. Once complete, it will create a `lora_model` folder.

## Phase 3: Running the App (Inference)

Once you have the `lora_model` folder, you can run the beautiful Streamlit UI.

1. **Activate your environment** (if not already activated):
   ```bash
   source venv/bin/activate
   ```

2. **Launch Streamlit**:
   ```bash
   streamlit run app.py
   ```

3. **Use the App**:
   Open your browser to the local URL provided by Streamlit (usually `http://localhost:8501`). Paste a contract and start scanning!
