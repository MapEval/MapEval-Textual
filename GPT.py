from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from LLM import LLM
import torch
from openai import AzureOpenAI
import json
from datetime import datetime



class GPT(LLM):
    def load_model(self):
        self.id = 6
        self.model = AzureOpenAI(
            api_version="2024-02-15-preview"
        )

                
    def generate(self, prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "You are an AI assistant that answers Place related MCQ questions."
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]
        
        completion = self.model.chat.completions.create(
            model="gpt-4o-08-06",  # model = "deployment_name"
            messages=messages,
            max_tokens=500,
            temperature=0,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )
        print(completion.choices[0].message.content)
        return completion.choices[0].message.content