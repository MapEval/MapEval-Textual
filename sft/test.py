from unsloth import FastLanguageModel
import json
import re
import json
import sys

def load_inference_results(file_path):
    with open(file_path, "r") as f:
        return json.load(f)
    
# Get the filename from the command line arguments
if len(sys.argv) < 2:
    print("Usage: python script.py <filename>")
    sys.exit(1)

filename = sys.argv[1]
# Load the model and tokenizer
output_dir = filename+"-finetuned"
# model = FastLanguageModel.from_pretrained(output_dir)
model, tokenizer = FastLanguageModel.from_pretrained(
    output_dir,  # Your saved model
    max_seq_length=6000,
    load_in_4bit=True,
)

# Check the tokenizer's max sequence length
max_seq_length = tokenizer.model_max_length
print(f"Max sequence length: {max_seq_length}")


def run_inference(example):
    prompt = (
        f"Context: {example['context']}\n"
        f"Question: {example['question']}\n"
        "Please choose the correct option and respond strictly with the option number (e.g., 0, 1, 2, 3, etc.):\n"
        "Options:\n"
    )

    if example["answer"]["correct"] == -1:
        prompt += "0. Unanswerable\n"

    for i, option in enumerate(example["answer"]["options"], start=1):
        if option.strip():
            prompt += f"{i}. {option}\n"

    prompt += f"Respond with just the option number ({'0, ' if example['answer']['correct'] == -1 else ''}1, 2, 3, etc.)."

    messages = [
        # {
        #     "role": "system",
        #     "content": "You are an AI assistant that answers Place related MCQ questions.",
        # },
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    model_inputs = tokenizer([text], return_tensors="pt").to("cuda")
    
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=10)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    
    # outputs = model.generate(
    #     input_ids=inputs, 
    #     max_new_tokens=1024, 
    #     use_cache=True, 
    #     temperature=1.5, 
    #     min_p=0.1,
    #     eos_token_id=terminators,
    # )
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
        0
    ]
    
    print(f"Generated response: {response}")

    # **Fix: Extract only the option number using regex**
    match = re.search(r"\b\d+\b", response)
    option_no = match.group(0) if match else "N/A"  # Default to "N/A" if no number is found

    classification = example.get("classification", "unanswerable")

    return {
        "input_text": prompt,
        "response": option_no,
        "classification": classification,
        "correct_option": str(example["answer"]["correct"] + 1)
    }

# Load the test dataset
def load_test_dataset(file_path):
    with open(file_path, "r", encoding='utf-8') as f:
        return json.load(f)

test_dataset = load_test_dataset("test.json")

# Run inference on all test examples
inference_results = []
for index, example in enumerate(test_dataset):
    print(f"Running inference for example {index}: {example['id']}")
    inference_results.append(run_inference(example))
    # break

# Save the results to a file
with open(filename+".json", "w", encoding='utf-8') as f:
    json.dump(inference_results, f, indent=4)

print("Inference completed. Results saved to json.")
