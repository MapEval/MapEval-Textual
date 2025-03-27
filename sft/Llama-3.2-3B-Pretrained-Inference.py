from unsloth import FastLanguageModel
import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load the model and tokenizer
output_dir = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(
        output_dir
    )
model = AutoModelForCausalLM.from_pretrained(
        output_dir, torch_dtype="auto", device_map="auto"
    )
# Check the tokenizer's max sequence length
max_seq_length = tokenizer.model_max_length
print(f"Max sequence length: {max_seq_length}")

def extract(response):
    try:
        print("Parsing:",response)
        match = re.search(r"\{.*?\}", response, re.DOTALL)
        print("Extracted json:", match.group(0))
        data = json.loads(match.group(0))
        chosen_option = data.get("option_no")
        explanation = data.get("explanation")
    except Exception as e:
        print(f"Error: {e}")
        chosen_option = None
        explanation = None
        
    print(f"Chosen option: {chosen_option}")
    return chosen_option


def run_inference(example):
    prompt = (
        "Context: "
        +  example["context"]
        + "Question: "
        + example["question"]
        + '''Please respond in the following JSON format:
{
    "option_no": <option index>, // "option_no" refers to the number corresponding to the chosen answer from the list of options. It should be between 1 and '''+str(len(example["answer"]["options"]))+'''.
    "explanation": "<reason>"
}

Example Prompt:
Question: What is the capital of France?
Option1: Berlin
Option2: Paris
Option3: Madrid
Option4: Rome

Example Response:
{
    "option_no": 2,
    "explanation": "Paris is the capital of France."
}

Provide your answer in this format.
'''
    ) 

    if example["classification"] is None:
        prompt += "Option0: Unanswerable, "

    for i in range(len(example["answer"]["options"])):
        if(example["answer"]["options"][i] == ""):
            break
        prompt = (
            prompt
            + "Option"
            + str(i + 1)
            + ": "
            + example["answer"]["options"][i]
            + ", "
        )

    messages = [
        {
            "role": "system",
            "content": "You are an AI assistant that answers Place related MCQ questions.",
        },
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    model_inputs = tokenizer([text], return_tensors="pt").to("cuda")
    
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=1024)
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
    option_no = extract(response)  # Default to "N/A" if no number is found
    option_no = option_no if option_no is not None else "N/A"

    classification = example.get("classification", "unanswerable")

    return {
        "input_text": prompt,
        "response": str(option_no),
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
with open("llama_pretrained_result.json", "w",  encoding='utf-8') as f:
    json.dump(inference_results, f, indent=4)

print("Inference completed. Results saved to json.")
