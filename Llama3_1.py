from LLM import LLM
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class Llama3_1(LLM):
    def load_model(self):
        self.id = 8
        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3.1-8B-Instruct"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3.1-8B-Instruct", torch_dtype="auto", device_map="auto"
        )
        self.model.eval()
        print("Llama3.1 model loaded")

    def generate(self, prompt: str) -> str:
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
