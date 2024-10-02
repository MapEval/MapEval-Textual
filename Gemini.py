from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from LLM import LLM
import torch

import google.generativeai as genai
from google.ai.generativelanguage import Part, Content
import json
from datetime import datetime



class Gemini(LLM):
    def load_model(self):
        self.id = 7
        genai.configure(api_key='AIzaSyB3MhiTdLd7KFC08sR-EBNjWO1M8ZNeYj8')
        generation_config = {
            "temperature": 0,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 500,
        }

        safety_settings = {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
        }

        self.model = genai.GenerativeModel(
            'gemini-pro', generation_config=generation_config, safety_settings=safety_settings)
                
    def generate(self, prompt: str) -> str:
        messages = [
            Content(
                parts=[
                    Part(
                            text="You are an AI assistant that answers Place related MCQ questions."
                    ),
                ],
                role="model"
            )
        ]
        chat = self.model.start_chat(history=messages)
        response = chat.send_message(prompt)
        print(response.text)
        return response.text