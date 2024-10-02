from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from LLM import LLM
import torch


class Mixtral(LLM):
    def load_model(self):
        self.id = 5
        self.tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mixtral-8x7B-Instruct-v0.1"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            torch_dtype="auto",
            device_map="auto",
        )
        self.model.eval()
        print("Mixtral model loaded")

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
            "max_new_tokens": 500,
            # "return_full_text": False,
            # "temperature": 0.0,
            "do_sample": False,
            "eos_token_id": terminators,
        }

        output = pipe(messages, **generation_args)
        return output[0]["generated_text"][-1]["content"]

        # inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to(
        #     "cuda"
        # )

        # outputs = self.model.generate(inputs, max_new_tokens=256)

        # print(outputs)
        # return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
