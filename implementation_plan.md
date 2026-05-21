# Legal "Red Flag" Scanner LLM Architecture

You are absolutely right. Training an LLM is only 20% of the battle. To have a real, usable product, we need a complete architecture: an interface, an inference engine, and a robust training pipeline. 

This plan outlines the complete, end-to-end architecture we will build for your Legal Scanner.

## User Review Required

> [!IMPORTANT]
> **WSL Requirement:** Because you are on an RTX 5050 (Windows), the training script **must** be run in WSL (Windows Subsystem for Linux) to use Unsloth. The UI and Inference can run on normal Windows, but training requires Linux. Are you comfortable setting up WSL for the training phase?

> [!NOTE]
> **UI Choice:** I am proposing we build the interface using **Streamlit**. It allows us to build a beautiful, modern web app in pure Python. Let me know if you prefer a different UI framework (like an API + React).

## Proposed Architecture

We will structure the project into several distinct, robust components:

### 1. Data Pipeline (Already Completed)
- **`prepare_data.py`**: Downloads and formats the CUAD dataset.
- **`legal_scanner_data.jsonl`**: The formatted training data.

### 2. The Training Engine
- **`train_gemma.py`**: A highly robust Unsloth training script. I will rewrite our previous draft to include:
  - Automatic memory management to prevent out-of-memory (OOM) bugs on your 6GB/8GB GPU.
  - Proper checkpoint saving (saving the model every X steps).
  - Merging the LoRA adapters to a final 16-bit model for inference.

### 3. The Inference Engine
- **`inference.py`**: This is the "brain" in production. It will load your fine-tuned model and expose a simple function: `scan_contract(contract_text, clause_type)`. It will handle text chunking if the contract is too long for Gemma's context window.

### 4. The User Interface (Streamlit)
- **`app.py`**: A clean, professional web interface where a user can:
  1. Paste their contract.
  2. Select which risks they want to scan for (e.g., "Non-compete", "Termination clause", "Unlimited Liability").
  3. View the extracted "Red Flags" highlighted by your AI.

### 5. Environment & Setup
- **`requirements.txt`**: The exact pip dependencies needed to run the app.
- **`README.md`**: Step-by-step instructions on how to install WSL, run the training, and launch the app.

## Proposed Changes

### Core Project
#### [MODIFY] [train_gemma.py](file:///C:/Users/majip/Downloads/xllm/legal_scanner_llm/train_gemma.py)
Update to include checkpointing, memory optimizations, and auto-merging of weights.

#### [NEW] [inference.py](file:///C:/Users/majip/Downloads/xllm/legal_scanner_llm/inference.py)
Create the production logic to load the model and process contracts.

#### [NEW] [app.py](file:///C:/Users/majip/Downloads/xllm/legal_scanner_llm/app.py)
Create the Streamlit web application.

#### [NEW] [requirements.txt](file:///C:/Users/majip/Downloads/xllm/legal_scanner_llm/requirements.txt)
List all dependencies.

#### [NEW] [README.md](file:///C:/Users/majip/Downloads/xllm/legal_scanner_llm/README.md)
Write documentation for the project.

## Verification Plan

### Manual Verification
1. I will write all the files to your local folder.
2. I will provide you with the exact command to install WSL and the dependencies.
3. You will run the Streamlit app (`streamlit run app.py`) to verify the UI looks good (even before the model is fully trained, we can load the base Gemma model to test the UI).
