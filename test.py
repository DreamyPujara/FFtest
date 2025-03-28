from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer
from transformers import BitsAndBytesConfig
from model import llm
from langchain_groq import ChatGroq
import torch
print(torch.cuda.is_available())  # Should return True for GPU usage

# Quantization config
quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

# Load your JSON dataset
import json
with open("DT Global Absence Accrual Matrix.json", "r") as f:
    data = json.load(f)
    print('loading json')

# Convert to Hugging Face Dataset
dataset = Dataset.from_dict({"formula": [d["formula"] for d in data], "prompt": [d["prompt"] for d in data]})
print('converted to hugging face dataset')

# Split into train and validation sets
dataset = dataset.train_test_split(test_size=0.2)
print('train test validation')


#load model 
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-Math-7B-Instruct"   # Replace with the actual model name
tokenizer = AutoTokenizer.from_pretrained(model_name)
print('loaded model successfully')

# Load the model with 8-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    quantization_config=quant_config,
    device_map="auto",
    use_flash_attention_2=True
    
)

model = prepare_model_for_kbit_training(model)

from peft import LoraConfig, get_peft_model

# Define LoRA configuration
lora_config = LoraConfig(
    r=8,  # Rank of the low-rank matrices
    lora_alpha=32,  # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Target attention layers
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

from transformers import TrainingArguments, Trainer

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,  # Reduced for stability
    gradient_accumulation_steps=4,
    optim="paged_adamw_8bit",
    learning_rate=2e-4,
    weight_decay=0.001,
    num_train_epochs=3,
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=50,
    fp16=False,  # Disable if using CPU
    bf16=torch.cuda.is_bf16_supported(),
    max_grad_norm=0.3,
    save_total_limit=3,
    report_to="none"
)

# Tokenize the dataset
def tokenize_function(examples):
    texts = [f"### Instruction: {p}\n### Response: {f}" 
            for p, f in zip(examples['prompt'], examples['formula'])]
            
    return tokenizer(
        texts,
        truncation=True,
        max_length=512,
        padding="max_length",
        add_special_tokens=True
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
)

# Fine-tune the model
trainer.train()

model.save_pretrained("./fine-tuned-llama")
tokenizer.save_pretrained("./fine-tuned-llama")