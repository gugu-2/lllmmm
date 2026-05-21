import os
import io
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pdfplumber
import docx2txt
import pytesseract
from PIL import Image

# Import our inference engine
# Note: In production, you'd only load this when needed or on startup.
try:
    from inference import LegalScannerEngine
    engine = LegalScannerEngine(model_path="lora_model")
except Exception as e:
    print("Warning: Could not load inference engine. Ensure model is trained.")
    engine = None

app = FastAPI(title="Legal Scanner API")

# Allow Next.js frontend to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_text(file_bytes: bytes, filename: str) -> str:
    extracted = ""
    filename = filename.lower()
    
    try:
        if filename.endswith('.pdf'):
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text: extracted += text + "\n"
        
        elif filename.endswith('.docx'):
            extracted = docx2txt.process(io.BytesIO(file_bytes))
            
        elif filename.endswith(('.png', '.jpg', '.jpeg')):
            image = Image.open(io.BytesIO(file_bytes))
            extracted = pytesseract.image_to_string(image)
            
        elif filename.endswith('.txt'):
            extracted = file_bytes.decode('utf-8')
            
        else:
            raise ValueError("Unsupported file type")
            
    except Exception as e:
        raise ValueError(f"Error extracting text: {str(e)}")
        
    return extracted

@app.post("/api/scan")
async def scan_contract(
    file: UploadFile = File(None), 
    contract_text: str = Form(None),
    clause_type: str = Form(...)
):
    if not engine or not engine.is_loaded:
        raise HTTPException(status_code=500, detail="AI Model is not loaded or trained yet.")
        
    text_to_scan = ""
    
    # Prioritize uploaded file
    if file and file.filename:
        file_bytes = await file.read()
        try:
            text_to_scan = extract_text(file_bytes, file.filename)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    elif contract_text:
        text_to_scan = contract_text
        
    if not text_to_scan.strip():
        raise HTTPException(status_code=400, detail="No contract text provided.")
        
    # Run the AI Inference
    try:
        result = engine.scan_contract(text_to_scan, clause_type)
        return {"status": "success", "clause": clause_type, "result": result, "extracted_text": text_to_scan[:1000] + "..." if len(text_to_scan) > 1000 else text_to_scan}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.get("/api/health")
def health_check():
    return {"status": "healthy", "model_loaded": engine is not None and engine.is_loaded}

# To run:
# uvicorn api:app --reload --port 8000
