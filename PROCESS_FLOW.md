# Legal Scanner: End-to-End Process Flow

The diagram below illustrates the complete lifecycle of the Legal "Red Flag" Scanner, from fetching the raw dataset to presenting the final AI analysis to the user.

## System Architecture Diagram

```mermaid
graph TD
    subgraph Data Pipeline
        A[Hugging Face Hub\n'chenghao/cuad_qa'] -->|Download Parquet| B(prepare_data.py)
        B -->|Format to Gemma Chat| C[(legal_scanner_data.jsonl)]
    end

    subgraph Training Pipeline WSL/Linux
        C -.->|Feed Training Data| D(train_gemma.py\nvia Unsloth)
        E[Base Model\nGoogle Gemma-2-9B-IT] -->|Load in 4-bit Quantization| D
        D -->|QLoRA Fine-tuning\nRank 16| F[(lora_model\nAdapters & Checkpoints)]
    end

    subgraph Inference & UI Production
        G[/User Pastes Contract/] --> H(app.py\nStreamlit Web UI)
        H -->|Request Clause Extraction| I(inference.py\nPyTorch Engine)
        
        E -.->|Load Base Model Base| I
        F -.->|Load Trained Weights| I
        
        I -->|Generate Strict Response| J[/AI Lawyer Analysis/]
        J -->|Display Extracted Risk| H
    end

    classDef script fill:#f9f,stroke:#333,stroke-width:2px;
    classDef data fill:#bbf,stroke:#333,stroke-width:2px;
    classDef model fill:#fbb,stroke:#333,stroke-width:2px;
    classDef ui fill:#bfb,stroke:#333,stroke-width:2px;

    class B,D,I script;
    class C,F data;
    class E model;
    class G,H,J ui;
```

## Step-by-Step Breakdown

1.  **Data Acquisition (`prepare_data.py`)**: 
    The script reaches out to Hugging Face and downloads the expert-annotated CUAD dataset. It automatically cleans and transforms the data into a `.jsonl` format specifically designed for Gemma's conversational training.

2.  **Model Fine-Tuning (`train_gemma.py`)**: 
    Using the Unsloth library in a Linux/WSL environment, the massive base Gemma model is shrunk down into 4-bit memory so it fits on your RTX 5050. The model is then trained on the `.jsonl` data, learning exactly how a lawyer extracts risky clauses.

3.  **Adapter Saving (`lora_model`)**: 
    Instead of saving a massive 18GB model, the system only saves the *differences* (the new knowledge) into a tiny LoRA adapter folder.

4.  **User Interaction (`app.py`)**: 
    A user opens the Streamlit web app, pastes in a new, unseen contract, and selects what they want to look for (e.g., "Termination clause").

5.  **Inference Execution (`inference.py`)**: 
    The engine quickly boots up the base model, snaps your trained LoRA adapter on top of it, and feeds the user's contract to the AI.

6.  **Results**: 
    The AI reads the contract, extracts the exact legal risk, and passes it back to the beautiful web interface for the user to read.
