from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from LLM import LLM
import torch
from openai import AzureOpenAI
import json
from datetime import datetime
import anthropic


class Claude(LLM):
    def load_model(self):
        self.id = 18
        self.model = anthropic.Anthropic()

                
    def generate(self, prompt: str) -> str:
        messages = [
            # {
            #     "role": "system",
            #     "content": "You are an AI assistant that answers Place related MCQ questions."
            # },
            {
                "role": "user",
                "content": prompt,
            },
        ]
        message = self.model.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=512,
            temperature=0,
            system="You are an AI assistant that answers Place related MCQ questions.",
            messages=messages
        )
        print(message.content[0].text)
        return message.content[0].text