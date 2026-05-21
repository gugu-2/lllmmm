# System Architecture

This document describes the technical architecture of the Legal "Red Flag" Scanner LLM application.

## Overview

The system is built as a modular AI application consisting of a data preparation pipeline, an offline local LLM training engine, an inference module, and a web-based user interface. Because of strict data privacy requirements in the legal and B2B sectors, all components are designed to run **locally and entirely offline**.

---

## 1. The Base Model
**Model Used:** Google `Gemma-2-9b-it` (Instruct)
*   **Why Gemma 2 9B?** It punches well above its weight class, beating many 70B models in reasoning and instruction following, while remaining small enough to run on consumer hardware.

---

## 2. Fine-Tuning Engine (Unsloth + QLoRA)
Because the target hardware (e.g., RTX 5050) has limited VRAM (6GB - 8GB), full-parameter fine-tuning is impossible. We utilize **Parameter-Efficient Fine-Tuning (PEFT)**.

*   **Unsloth Framework:** Unsloth rewrites PyTorch kernels in Triton, resulting in 2x faster training and 70% less memory usage.
*   **4-bit Quantization:** The base model is loaded in 4-bit precision via `bitsandbytes`, drastically reducing its memory footprint from ~18GB to ~6GB.
*   **LoRA (Low-Rank Adaptation):** We freeze the base model and only train a small matrix (Rank = 16) applied to specific attention modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`, etc.). This results in a tiny, easily portable adapter file rather than a massive new model.

---

## 3. Data Pipeline
The model is trained on the **CUAD (Contract Understanding Atticus Dataset)**.
*   **Format Conversion:** `prepare_data.py` downloads the Parquet format dataset from Hugging Face and converts the SQuAD-like QA format into a generative chat format suitable for Gemma's `user/model` turn template.
*   **Output:** A `.jsonl` file where every line is a perfect prompt-response pair representing a lawyer extracting a risky clause.

---

## 4. Inference Engine (`inference.py`)
In production, the inference engine is responsible for interacting with the AI.
*   **Adapter Loading:** It loads the base Gemma model and applies the custom `lora_model` weights on top.
*   **FastLanguageModel:** It switches the model to inference mode (`FastLanguageModel.for_inference()`), which enables 2x faster generation by disabling autograd and optimizing attention caches.
*   **Text Generation:** Uses highly constrained generation parameters (e.g., `temperature=0.1`) to prevent hallucinations and force the model to stick strictly to the contract text.

---

## 5. Web Interface (`app.py`)
Built using **Streamlit**.
*   **State Management:** The LLM engine is cached in memory (`@st.cache_resource`) so it only loads into the GPU once when the app starts, rather than every time the user clicks "Scan".
*   **User Flow:** A split-pane design where users input their contract on the left, select their desired target clause from a dropdown on the right, and view the AI's extraction results instantly.
