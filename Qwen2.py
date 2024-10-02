from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from LLM import LLM
import torch


class Qwen2(LLM):
    def load_model(self):
        self.id = 3
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
        self.model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2-7B-Instruct", torch_dtype="auto", device_map="auto"
        )
        self.model.eval()
        print("Qwen2 model loaded")

    def generate(self, prompt: str) -> str:
        # inputs = self.tokenizer(prompt, return_tensors="pt")
        # outputs = self.model.generate(**inputs)
        # return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        device = "cuda"
        messages = [
            {
                "role": "system",
                "content": "You are an AI assistant that answers Place related MCQ questions.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(device)

        generated_ids = self.model.generate(model_inputs.input_ids, max_new_tokens=4096)
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ]
        return response
