'use client';

import { useState, useRef } from 'react';
import { UploadCloud, FileText, AlertTriangle, ShieldCheck, FileCheck } from 'lucide-react';

export default function Home() {
  const [inputMode, setInputMode] = useState<'upload' | 'text'>('upload');
  const [file, setFile] = useState<File | null>(null);
  const [contractText, setContractText] = useState('');
  const [clause, setClause] = useState('Termination clause');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<{status: string, clause: string, result: string, extracted_text?: string} | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
    }
  };

  const handleScan = async () => {
    if (inputMode === 'upload' && !file) {
      setError("Please upload a contract document first.");
      return;
    }
    if (inputMode === 'text' && !contractText.trim()) {
      setError("Please paste contract text first.");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    if (inputMode === 'upload' && file) {
      formData.append('file', file);
    } else if (inputMode === 'text' && contractText) {
      formData.append('contract_text', contractText);
    }
    formData.append('clause_type', clause);

    try {
      // Connect to the local FastAPI backend running the AI Model
      const response = await fetch('http://localhost:8000/api/scan', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'An error occurred during scanning.');
      }

      setResult(data);
    } catch (err: any) {
      setError(err.message || "Failed to connect to the AI engine. Ensure the FastAPI backend is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="container animate-fade-in">
      <header className="app-header">
        <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '1.5rem' }}>
          <div className="icon-badge">
             <ShieldCheck size={36} />
          </div>
        </div>
        <h1>Legal Risk Scanner</h1>
        <p>Upload a contract (PDF, Word, or Image) or paste its text, and our specialized AI will instantly identify and extract critical clauses.</p>
      </header>

      <div className="main-grid">
        {/* Left Column: Upload & Config */}
        <div className="card">
          <h2>1. Input Contract</h2>
          
          <div className="tabs-container">
            <button 
              className={`tab-btn ${inputMode === 'upload' ? 'active' : ''}`}
              onClick={() => setInputMode('upload')}
            >
              Upload File
            </button>
            <button 
              className={`tab-btn ${inputMode === 'text' ? 'active' : ''}`}
              onClick={() => setInputMode('text')}
            >
              Paste Raw Text
            </button>
          </div>

          {inputMode === 'upload' ? (
            <div 
              className={`dropzone ${file ? 'has-file' : ''}`}
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <input 
                type="file" 
                ref={fileInputRef} 
                onChange={handleFileChange} 
                style={{ display: 'none' }} 
                accept=".pdf,.docx,.txt,.png,.jpg,.jpeg"
              />
              {file ? (
                <>
                  <FileCheck className="dropzone-icon success-icon animate-bounce-subtle" />
                  <div>
                    <p className="file-name">{file.name}</p>
                    <p className="file-size">{(file.size / 1024).toFixed(1)} KB • Ready to scan</p>
                  </div>
                </>
              ) : (
                <>
                  <UploadCloud className="dropzone-icon animate-pulse-slow" />
                  <div>
                    <p className="dropzone-text">Click to upload or drag & drop</p>
                    <p className="dropzone-subtext">PDF, DOCX, TXT, PNG, or JPG</p>
                  </div>
                </>
              )}
            </div>
          ) : (
            <div className="textarea-container">
              <textarea
                className="text-input-field"
                placeholder="Paste your contract terms, agreements, or clauses here..."
                value={contractText}
                onChange={(e) => setContractText(e.target.value)}
                rows={8}
              />
            </div>
          )}

          <div style={{ marginTop: '2.5rem' }}>
            <h2>2. Target Risk Clause</h2>
            <div className="input-group">
              <label className="input-label">Select Clause to Extract</label>
              <div className="select-wrapper">
                <select 
                  className="select-input"
                  value={clause}
                  onChange={(e) => setClause(e.target.value)}
                >
                  <option value="Termination clause">Termination Clause</option>
                  <option value="Non-compete clause">Non-compete Clause</option>
                  <option value="Governing Law">Governing Law</option>
                  <option value="Indemnification">Indemnification</option>
                  <option value="Liability Cap">Liability Cap</option>
                  <option value="Audit Rights">Audit Rights</option>
                  <option value="Exclusivity">Exclusivity</option>
                  <option value="Document Name">Document Name</option>
                </select>
              </div>
            </div>

            <button 
              className="btn-primary" 
              onClick={handleScan}
              disabled={loading || (inputMode === 'upload' ? !file : !contractText.trim())}
            >
              {loading ? (
                <><div className="loader"></div> Analyzing Contract...</>
              ) : (
                <><AlertTriangle size={20} /> Analyze Contract</>
              )}
            </button>
            
            {error && (
              <div className="error-banner animate-shake">
                <strong>Analysis Failed:</strong> {error}
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Results */}
        <div className="card result-card" style={{ display: 'flex', flexDirection: 'column' }}>
          <h2>Analysis Results</h2>
          
          {result ? (
            <div className="animate-fade-in" style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                <span className="tag">{result.clause}</span>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <ShieldCheck size={18} color="var(--success)" />
                  <span style={{ fontWeight: 600, color: 'var(--success)', fontSize: '0.9rem' }}>Scan Complete</span>
                </div>
              </div>
              
              <div className="result-box glow-success">
                {result.result === "No relevant clause found." ? (
                  <span style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>The AI could not find any clause matching this criteria in the document.</span>
                ) : (
                  result.result
                )}
              </div>
              
              {result.extracted_text && (
                 <div style={{ marginTop: '2.5rem' }}>
                   <p className="preview-label">Extracted Text Preview:</p>
                   <div className="preview-box">
                     {result.extracted_text}
                   </div>
                 </div>
              )}
            </div>
          ) : (
            <div className="empty-results">
              <FileText size={56} className="empty-icon animate-pulse-slow" />
              <p>Ready to analyze. Provide a contract on the left and start the scan to extract AI legal insights here.</p>
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
