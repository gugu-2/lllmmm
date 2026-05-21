import torch
from unsloth import FastLanguageModel
import textwrap

class LegalScannerEngine:
    def __init__(self, model_path="lora_model"):
        self.max_seq_length = 2048
        dtype = None
        load_in_4bit = True
        
        print(f"Loading model and adapters from {model_path}...")
        
        try:
            # Unsloth handles loading the base model and adapters in a single optimized pass
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name = model_path,
                max_seq_length = self.max_seq_length,
                dtype = dtype,
                load_in_4bit = load_in_4bit,
            )
            # Enable inference mode to run 2x faster
            FastLanguageModel.for_inference(self.model)
            self.is_loaded = True
            print("Model successfully loaded for inference!")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.is_loaded = False
            self.model = None
            self.tokenizer = None

    def scan_contract(self, contract_text, clause_type):
        if not self.is_loaded:
            return "Error: Model not loaded. Did you train it first?"

        # We must use the exact prompt format we used during training
        prompt = f"<start_of_turn>user\nYou are an expert corporate lawyer. Review the following contract text and extract any clauses related to: {clause_type}\n\nContract Text:\n{contract_text}<end_of_turn>\n<start_of_turn>model\n"

        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
        prompt_len = inputs.input_ids.shape[1]

        print("Generating response...")
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=256, 
            use_cache=True,
            temperature=0.1, # Keep it low so the lawyer doesn't hallucinate
        )
        
        # Extract just the model's new tokens to avoid fragile string split on "model\n"
        generated_tokens = outputs[0][prompt_len:]
        
        # Decode the generated answer
        answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        return answer

# Simple test if run directly
if __name__ == "__main__":
    engine = LegalScannerEngine()
    test_contract = "This Agreement shall be governed by the laws of the State of California, without regard to its conflict of laws principles."
    result = engine.scan_contract(test_contract, "Governing Law")
    print("\n--- Result ---")
    print(result)
