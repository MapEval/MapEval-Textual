from unsloth import FastLanguageModel
import torch
from datasets import load_dataset, Dataset
import json

def format_example(example):
    prompt = (
        f"Context: {example['context']}\n"
        f"Question: {example['question']}\n"
        "Please choose the correct option and respond strictly with the option number (e.g., 0, 1, 2, 3, etc.):\n"
        "Options:\n"
    )

    # Add 'Unanswerable' option if applicable
    # if example["answer"]["correct"] == -1:
    prompt += "0. Unanswerable\n"

    # Append available options
    for i, option in enumerate(example["answer"]["options"], start=1):
        if option.strip():  # Ensuring no empty options
            prompt += f"{i}. {option}\n"

    prompt += f"Respond with just the option number (0, 1, 2, 3, etc.)."

    correct_option = example["answer"]["correct"] + 1  # Correct option number
    return {"input_text": prompt, "target_text": str(correct_option)}

# Load the training dataset
file_path = 'train.json'

# Read the JSON file
with open(file_path, 'r', encoding='utf-8') as file:
    raw_data = json.load(file)

print(f"Loaded {len(raw_data)} items from {file_path}")

# Convert to Hugging Face Dataset
dataset = Dataset.from_list([format_example(d) for d in raw_data])
print("Dataset prepared successfully.")

def convert_to_conversations(example):
    return {
        "conversations": [
            {"role": "user", "content": example["input_text"]},
            {"role": "assistant", "content": example["target_text"]},
        ]
    }

# Apply transformation
dataset = dataset.map(convert_to_conversations)

# Verify structure
# print(dataset[0])  # Check a sample


max_seq_length = 4096
dtype = None # Auto detection of dtype
load_in_4bit = True # Reduce memory usage with 4-bit quantization
model, tokenizer = FastLanguageModel.from_pretrained(
 model_name = "unsloth/Llama-3.2-3B-Instruct",
 max_seq_length = max_seq_length,
 dtype = dtype,
 load_in_4bit = load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
 model,
 r = 16, # LoRA rank
 target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
 lora_alpha = 16,
 lora_dropout = 0, # Optimized at 0
 bias = "none", # No additional bias terms
 use_gradient_checkpointing = "unsloth", # Gradient checkpointing to save memory
 random_state = 3407,
 use_rslora = False, # Rank stabilized LoRA, can be enabled for stability
)



def formatting_prompts_func(examples):
 convos = examples["conversations"]
 texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
 return { "text": texts }
dataset = dataset.map(formatting_prompts_func, batched=True)

# Verify structure
print(dataset[0])  # Check a sample

from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported
trainer = SFTTrainer(
 model = model,
 tokenizer = tokenizer,
 train_dataset = dataset,
 dataset_text_field = "text",
 max_seq_length = max_seq_length,
 data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
 dataset_num_proc = 2,
 packing = False,
 args = TrainingArguments(
 per_device_train_batch_size = 2,
 gradient_accumulation_steps = 4,
 warmup_steps = 5,
 max_steps = 60,
 learning_rate = 2e-4,
 fp16 = not is_bfloat16_supported(),
 bf16 = is_bfloat16_supported(),
 logging_steps = 1,
 optim = "adamw_8bit",
 weight_decay = 0.01,
 lr_scheduler_type = "linear",
 seed = 3407,
 output_dir = "outputs",
 ),
)


from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
 trainer,
 instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
 response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
)
trainer_stats = trainer.train()

# Save the model and tokenizer
output_dir = "llama-3.2-3b-finetuned"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
