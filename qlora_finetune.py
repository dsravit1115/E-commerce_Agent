from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import torch
import pandas as pd
from datasets import Dataset

def qlora_finetune_stub():
    print(" Starting QLoRA fine-tuning (stub for interview demo)...")

    # Sample dataset
    data = {
        "text": [
            "Customer from North region purchased items worth $1200.",
            "Sales in South region increased by 20% compared to last month."
        ]
    }
    dataset = Dataset.from_pandas(pd.DataFrame(data))

    # Load tokenizer & model
    model_name = "tiiuae/falcon-rw-1b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map="auto")
    model = prepare_model_for_kbit_training(model)

    # Apply LoRA
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query_key_value"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)

    # Tokenize
    def tokenize(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

    tokenized = dataset.map(tokenize)

    # Training args
    args = TrainingArguments(
        output_dir="./qlora-finetune-output",
        per_device_train_batch_size=1,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=10,
        save_total_limit=2,
        report_to="none"
    )

    trainer = Trainer(model=model, args=args, train_dataset=tokenized)
    print("Running Trainer...")
    trainer.train()
    print(" Fine-tuning complete (mock run).")
