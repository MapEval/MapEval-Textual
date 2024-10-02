from BenchmarkDataset import BenchmarkDataset
from transformers import Trainer, TrainingArguments
from LLM import LLM
import requests
import time
import re
import json
# def extract(res):
#     match = re.search(r"\^\^(.*?)\^\^", res)
#     if match:
#         s = match.group(1)
#         for char in s:
#             if char.isdigit():
#                 return char
#     else:
#         # Search fo Option%d
#         match = re.search(r"Option(\d)", res)
#         if match:
#             s = match.group(1)
#             for char in s:
#                 if char.isdigit():
#                     return char
#         # Search for Option %d pre-suf with space
#         # else:
#         #     match = re.search(r" (\d) ", res)
            
#         #     if match:
#         #         s = match.group(1)
#         #         for char in s:
#         #             if char.isdigit():
#         #                 return char
    
#     print("***ERROR: Could not extract option from response.***", res)   
#     return None  # Return None if no numeric character is found

def extract(response):
    try:
        print("Parsing:",response)
        match = re.search(r"\{.*?\}", response, re.DOTALL)
        print("Extracted json:", match.group(0))
        data = json.loads(match.group(0))
        chosen_option = data.get("option_no")
        explanation = data.get("explanation")
    except Exception as e:
        print(f"Error: {e}")
        chosen_option = None
        explanation = None
    return chosen_option

def search_evaluation_by_model(data, model_id, type):
    evaluations = data.get("evaluation", [])
    for evaluation in evaluations:
        if evaluation.get("model_id") == model_id and evaluation.get("type") == type:
            return evaluation
    return None

class Evaluator:
    def __init__(self, model: LLM, dataset: BenchmarkDataset, type: int = 0):
        self.model = model
        self.dataset = dataset
        self.results = []
        self.type = type

    def evaluate(self):
        self.model.load_model()
        data = self.dataset.load_data()

        list=[]
        for i in range(0, len(data)):
            item = data[i]
            print("Running", i + 1, "/", len(data), ":", item["id"])
            if item["context"] == "" or (self.type == 2 and item["context_gpt"] == ""):
                self.results.append(
                    {
                        "prompt": "",
                        "id": item["id"],
                    }
                )
                continue
            
            if(search_evaluation_by_model(item, self.model.id, self.type) or item["classification"] is not None):
                print("Already evaluated")
                continue
            
            # prompt = (
            #     "Context: "
            #     + item["context"]
            #     + "Question: "
            #     + item["question"]
            #     + "(Remember to answer the question strictly based on the given context, without using any external knowledge or assumptions.)"
            # )

            prompt = (
                "Context: "
                + item["context"]
                + "Question: "
                + item["question"]
                + '''Please respond in the following JSON format:
{
  "option_no": <option index>, // "option_no" refers to the number corresponding to the chosen answer from the list of options. It should be between 1 and '''+str(len(item["answer"]["options"]))+'''. If there is no correct answer in the options, then set "option_no" to 0.
  "explanation": "<reason>"
}

Example Prompt:
Question: What is the capital of France?
Option0: Unanswerable
Option1: Berlin
Option2: Paris
Option3: Madrid
Option4: Rome

Example Response:
{
  "option_no": 2,
  "explanation": "Paris is the capital of France."
}

Provide your answer in this format. Remember to answer the question strictly based on the given context, without using any external knowledge or assumptions. 
'''
            )

#             prompt = (
#                 "Context: "
#                 +  (str(item["context_json"]) if self.type == 1 else str(item["context_gpt"]) if self.type == 2 else item["context"])
#                 + "Question: "
#                 + item["question"]
#                 + '''Please respond in the following JSON format:
# {
#   "option_no": <option index>, // "option_no" refers to the number corresponding to the chosen answer from the list of options. It should be between 1 and '''+str(len(item["answer"]["options"]))+'''.
#   "explanation": "<reason>"
# }

# Example Prompt:
# Question: What is the capital of France?
# Option1: Berlin
# Option2: Paris
# Option3: Madrid
# Option4: Rome

# Example Response:
# {
#   "option_no": 2,
#   "explanation": "Paris is the capital of France."
# }

# Provide your answer in this format. Remember to answer the question strictly based on the given context, without using any external knowledge or assumptions. 
# '''
#             ) 

            prompt += "Option0: Unanswerable, "

            for i in range(len(item["answer"]["options"])):
                if(item["answer"]["options"][i] == ""):
                    break
                prompt = (
                    prompt
                    + "Option"
                    + str(i + 1)
                    + ": "
                    + item["answer"]["options"][i]
                    + ", "
                )


            if type == 3:
                prompt = prompt + " Let's think step by step. "

            print("Prompt is created. Now passing to the model.")
            
            count = 0
            ans = None
            while True:
                try:
                    response = self.model.generate(prompt)
                    print("Response: ",response)
                    tmp = extract(response)
                    if tmp and tmp > len(item["answer"]["options"]) and count < 3:
                        print("Error: The model generated an invalid response.")
                        print(response)
                        count += 1
                    # if not extract(response) and count < 2:
                    #     
                    else:
                        ans = tmp
                        break
                except Exception as e:
                    print("Error: The model could not generate a response.")
                    print(e)
                    count += 1
                    if len(list) > 0:
                        response = requests.post(
                            "http://localhost:5000/api/evaluation/", json=list
                        )
                    # clear list
                    list = []
                    time.sleep(10)  # Sleep for 10 seconds before retrying
                    if count == 5:
                        response = None
                        break
                    continue
            
            # print(response, tmp)
            if response is None:
                print("Error: Skipping query.")
                continue
            
            
            
            
            try:
                result =  {
                        "id": item["id"],
                        "prompt": prompt,
                        "response": response,
                        "ground_truth": item["answer"]["correct"] + 1,
                        # "data": item,
                    }
                self.results.append(result)
                
                if result["prompt"] == "":
                    list.append(
                        {
                            "query_id": result["id"],
                            "model_id": self.model.id,
                            "answer": "",
                            "verdict": "invalid",
                            "type": self.type,
                            "option": 0,
                        }
                    )
                else:
                    try:
                        option = extract(result["response"])
                        response = int(option)
                        if response == result["ground_truth"]:
                            list.append(
                                {
                                    "query_id": result["id"],
                                    "model_id": self.model.id,
                                    "answer": result["response"],
                                    "verdict": "right",
                                    "type": self.type,
                                    "option": response
                                }
                            )
                        elif response == 0:
                            list.append(
                                {
                                    "query_id": result["id"],
                                    "model_id": self.model.id,
                                    "answer": result["response"],
                                    "verdict": "invalid",
                                    "type": self.type,
                                    "option": response
                                }
                            )
                        else:
                            list.append(
                                {
                                    "query_id": result["id"],
                                    "model_id": self.model.id,
                                    "answer": result["response"],
                                    "verdict": "wrong",
                                    "type": self.type,
                                    "option": response
                                }
                            )

                    except Exception:
                        list.append(
                            {
                                "query_id": result["id"],
                                "model_id": self.model.id,
                                "answer": result["response"],
                                "verdict": "invalid",
                                "type": self.type,
                                "option": 0
                            }
                        )

   
            except ValueError:
                print("Error: The response could not be converted to an integer.")
                
            if len(list) >= 10:
                print(list[-1])
                response = requests.post(
                    "http://localhost:5000/api/evaluation/", json=list
                )
                # print(response)
                list = []
                time.sleep(10)  # Sleep for 10 seconds before retrying
            # break
        
        if len(list) > 0:
            response = requests.post(
                "http://localhost:5000/api/evaluation/", json=list
            )
            # print(response)
            list = []
            # time.sleep(10)

    def compute_metrics(self):
        correct_answers = 0
        total_questions = len(self.results)
        invalid_questions = 0
        invalid_answers = 0

        list = []

        for result in self.results:
            # print(result)
            if result["prompt"] == "":
                invalid_questions += 1
                # print(result)
                list.append(
                    {
                        "query_id": result["id"],
                        "model_id": self.model.id,
                        "answer": "",
                        "verdict": "invalid",
                        "type": self.type,
                    }
                )
            else:
                try:
                    option = extract(result["response"])
                    # response = int(result["response"].split()[0].strip(":.")[-1])
                    response = int(option)
                    if response == result["ground_truth"]:
                        correct_answers += 1
                        list.append(
                            {
                                "query_id": result["id"],
                                "model_id": self.model.id,
                                "answer": result["response"],
                                "verdict": "right",
                                "type": self.type,
                            }
                        )
                    elif response == 0:
                        invalid_answers += 1
                        list.append(
                            {
                                "query_id": result["id"],
                                "model_id": self.model.id,
                                "answer": result["response"],
                                "verdict": "invalid",
                                "type": self.type,
                            }
                        )
                    else:
                        list.append(
                            {
                                "query_id": result["id"],
                                "model_id": self.model.id,
                                "answer": result["response"],
                                "verdict": "wrong",
                                "type": self.type,
                            }
                        )

                except Exception:
                    # print("Error: The response could not be converted to an integer.")
                    invalid_answers += 1
                    list.append(
                        {
                            "query_id": result["id"],
                            "model_id": self.model.id,
                            "answer": result["response"],
                            "verdict": "invalid",
                            "type": self.type,
                        }
                    )
        # print(list)
        # print(response)
        accuracy = correct_answers * 100 / (total_questions - invalid_questions)
        accuracy = "{:.2f}".format(accuracy)

        # Open the file in write mode ('w')
        print(f"Accuracy: {accuracy}%\n")
        print(f"{invalid_questions} invalid questions\n")
        print(f"{invalid_answers} invalid responses\n")

    def print_results(self):
        # for result in self.results:
        # print(f"Prompt: {result['prompt']}")
        # print(f"Response: {result['response']}")
        # print(f"Ground Truth: {result['ground_truth']}\n")
        pass
