import os
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# Configuration
max_seq_length = 2048 # Max context window. Decrease to 1024 if you get Out Of Memory errors.
dtype = None # Auto detection
load_in_4bit = True # 4bit quantization is MUST for RTX 5050

def main():
    print("Loading Gemma 2 9B model in 4-bit quantization...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/gemma-2-9b-it", 
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    print("Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Rank
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )

    print("Loading your custom legal scanner dataset...")
    # Load the JSONL file we generated earlier
    dataset = load_dataset("json", data_files="legal_scanner_data.jsonl", split="train")

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False,
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 10,
            max_steps = 500, # Run for 500 steps. You can increase this later.
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 10,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            save_steps = 100, # Save checkpoint every 100 steps
        ),
    )

    print("Starting Training! This will take time based on your GPU...")
    trainer_stats = trainer.train()

    print("Training Complete! Saving the final model...")
    # Save the LoRA adapters locally
    model.save_pretrained("lora_model")
    tokenizer.save_pretrained("lora_model")
    print("Model saved to 'lora_model' directory. You are ready for inference!")

if __name__ == "__main__":
    main()
