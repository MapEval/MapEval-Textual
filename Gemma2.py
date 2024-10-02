from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from LLM import LLM
import torch


class Gemma2(LLM):
    def load_model(self):
        self.id = 15
        self.tokenizer = AutoTokenizer.from_pretrained(
            "google/gemma-2-9b-it"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2-9b-it",
            torch_dtype="auto",
            device_map="auto",
        )
        self.model.eval()
        print("Gemma2.0 model loaded")

    def generate(self, prompt: str) -> str:
        messages = [
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

        terminators = [
            pipe.tokenizer.eos_token_id,
            pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        generation_args = {
            "max_new_tokens": 512,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
            "eos_token_id": terminators,
        }

        output = pipe(messages, **generation_args)
        print(output)
        return output[0]["generated_text"]