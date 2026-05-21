import os
from datasets import load_dataset
import pandas as pd

def main():
    print("Downloading the CUAD dataset from Hugging Face...")
    # Using a parquet version of CUAD since the original script is deprecated
    dataset = load_dataset("chenghao/cuad_qa")
    
    train_data = dataset["train"]
    formatted_examples = []
    
    print("Formatting data for Gemma...")
    for item in train_data:
        context = item['context']
        question = item['question']
        
        answers = item['answers']['text']
        answer_text = answers[0] if len(answers) > 0 else "No relevant clause found."
        
        prompt = f"<start_of_turn>user\nYou are an expert corporate lawyer. Review the following contract text and extract any clauses related to: {question}\n\nContract Text:\n{context}<end_of_turn>\n<start_of_turn>model\n{answer_text}<end_of_turn>"
        
        formatted_examples.append({"text": prompt})
    
    df = pd.DataFrame(formatted_examples)
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "legal_scanner_data.jsonl")
    df.to_json(output_path, orient="records", lines=True)
    
    print(f"\nSuccess! Saved {len(df)} training examples to 'legal_scanner_data.jsonl'")
    print("\n--- Sneak Peek of Example 1 ---")
    print(df['text'].iloc[0][:300] + "...\n")

if __name__ == "__main__":
    main()
