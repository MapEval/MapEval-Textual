from LLM import LLM
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from groq import Groq

class Llama3_1_70B(LLM):
    def load_model(self):
        self.id = 21
        self.model = Groq()
        print("Llama3.1-70B model loaded")

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
        
        
        chat_completion = self.model.chat.completions.create(
            messages=messages,
            model="llama-3.1-70b-versatile",
            temperature=0.0,
            max_tokens=512,
            top_p=1,
            stop=None,
            stream=False,
            # do_sample=False,
        )


        return chat_completion.choices[0].message.content
