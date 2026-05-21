import streamlit as st
import time
import io
import pdfplumber
import docx2txt
import pytesseract
from PIL import Image
from inference import LegalScannerEngine

# Page config
st.set_page_config(page_title="AI Legal Red Flag Scanner", page_icon="⚖️", layout="wide")

# Title and Description
st.title("⚖️ AI Legal Red Flag Scanner")
st.markdown("Powered by a custom fine-tuned Gemma 2 LLM to identify critical legal risks in contracts.")

# Helper to extract text
def extract_text(uploaded_file):
    if uploaded_file is None:
        return ""
    
    file_name = uploaded_file.name.lower()
    extracted = ""
    
    try:
        if file_name.endswith('.pdf'):
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text: extracted += text + "\n"
        
        elif file_name.endswith('.docx'):
            extracted = docx2txt.process(uploaded_file)
            
        elif file_name.endswith(('.png', '.jpg', '.jpeg')):
            image = Image.open(uploaded_file)
            extracted = pytesseract.image_to_string(image)
            
        elif file_name.endswith('.txt'):
            extracted = uploaded_file.read().decode('utf-8')
            
        else:
            st.error("Unsupported file type!")
            
    except Exception as e:
        st.error(f"Error extracting text: {e}")
        
    return extracted

import os

# Initialize the inference engine only once if it exists
@st.cache_resource(show_spinner="Loading AI Legal Engine...")
def load_engine_cached():
    return LegalScannerEngine(model_path="lora_model")

# Only invoke cache if model is ready, preventing caching of a failed engine state
if os.path.exists("lora_model"):
    engine = load_engine_cached()
else:
    class DummyEngine:
        def __init__(self):
            self.is_loaded = False
        def scan_contract(self, contract_text, clause_type):
            return "Error: AI Model is not loaded or trained yet."
    engine = DummyEngine()

if not engine.is_loaded:
    st.error("Model not found! Have you run `train_gemma.py` in WSL yet to create the 'lora_model' folder?")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Contract Input")
    
    # NEW: File Uploader
    uploaded_file = st.file_uploader("Upload a PDF, Word Doc, or Image", type=["pdf", "docx", "txt", "png", "jpg", "jpeg"])
    
    # Automatically extract text if a file is uploaded
    default_text = ""
    if uploaded_file is not None:
        with st.spinner("Extracting text from document..."):
            default_text = extract_text(uploaded_file)
            if default_text:
                st.success("Text extracted successfully!")
    
    contract_text = st.text_area("Or paste your contract here:", value=default_text, height=400, placeholder="THIS AGREEMENT is entered into...")

with col2:
    st.subheader("Scan Configuration")
    clause_type = st.selectbox(
        "What are you looking for?",
        (
            "Termination clause",
            "Non-compete clause",
            "Governing Law",
            "Indemnification",
            "Liability Cap",
            "Audit Rights",
            "Exclusivity",
            "Document Name"
        )
    )
    
    if st.button("Scan Contract", type="primary", use_container_width=True):
        if not contract_text.strip():
            st.warning("Please provide a contract (upload or paste) first.")
        elif not engine.is_loaded:
            st.error("Cannot scan. Model is not trained/loaded.")
        else:
            with st.spinner("AI Lawyer is reviewing the contract..."):
                start_time = time.time()
                result = engine.scan_contract(contract_text, clause_type)
                end_time = time.time()
                
            st.success(f"Scan completed in {round(end_time - start_time, 2)}s")
            
            st.subheader("Extraction Result:")
            st.info(result)
