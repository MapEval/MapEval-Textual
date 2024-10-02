from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from LLM import LLM
import torch


class MistralNemo(LLM):
    def load_model(self):
        self.id = 12
        self.tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-Nemo-Instruct-2407"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-Nemo-Instruct-2407", torch_dtype="auto", device_map="auto"
        )
        self.model.eval()
        print("Mistral Nemo model loaded")

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
            # "return_full_text": False,
            # "temperature": 0.0,
            "do_sample": False,
            "eos_token_id": terminators,
        }

        output = pipe(messages, **generation_args)
        return output[0]["generated_text"][-1]["content"]
