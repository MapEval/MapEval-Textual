from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from LLM import LLM
import torch


class Phi3(LLM):
    def load_model(self):
        self.id = 1
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/Phi-3-mini-128k-instruct"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-128k-instruct",
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        print("Phi3 model loaded")

    def generate(self, prompt: str) -> str:
        # inputs = self.tokenizer(prompt, return_tensors="pt")
        # outputs = self.model.generate(**inputs)
        # return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

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

        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

        generation_args = {
            "max_new_tokens": 512,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
        }

        output = pipe(messages, **generation_args)

        return output[0]["generated_text"]
