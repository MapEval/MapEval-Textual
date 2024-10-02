from BenchmarkDataset import BenchmarkDataset
from Evaluator import Evaluator
from Llama3 import Llama3
from Mistral import Mistral
from Phi3 import Phi3
from Qwen2 import Qwen2
from Mixtral import Mixtral
from Gemini import Gemini
from GPT import GPT
from GPT_4o_mini import GPT_4o_mini
from Llama3_1 import Llama3_1
from Llama3_2 import Llama3_2
from Gemini1_5 import Gemini1_5
from Phi3Medium import Phi3Medium
from Phi3_5 import Phi3_5
from MistralNemo import MistralNemo
from Gemini1_5_Flash import Gemini1_5_Flash
from Gemma2 import Gemma2
from ChatGPT import ChatGPT
from Claude import Claude
from Phi3_4k import Phi3_4k
from Llama3_1_70B import Llama3_1_70B
from Llama3_2_90B import Llama3_2_90B
from GPT4 import GPT4
import torch
import argparse

# def main():
#     # Load and preprocess dataset
#     dataset = BenchmarkDataset(filepath="dataset.json")
#     dataset.preprocess_data()

#     # Initialize models
#     models = [Qwen2(), Phi3(), Mistral(), Llama3()]

#     # Evaluate each model
#     for model in models:
#         print(f"Evaluating model: {model.__class__.__name__}")
#         evaluator = Evaluator(model=model, dataset=dataset)
#         evaluator.evaluate()
#         evaluator.print_results()
#         print(model.__class__.__name__, "metrics")
#         evaluator.compute_metrics()
#         torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model.")
    parser.add_argument("model", type=str,
                        help="Name of the model to evaluate")
    args = parser.parse_args()

    # Load and preprocess dataset
    dataset = BenchmarkDataset(filepath="dataset.json")
    dataset.preprocess_data()

    models = [Qwen2(), Phi3(), Mistral(), Llama3()]
    # Initialize models
    if args.model == "Phi3":
        model = Phi3()
    elif args.model == "Phi3Medium":
        model = Phi3Medium()
    elif args.model == "Mistral":
        model = Mistral()
    elif args.model == "Llama3":
        model = Llama3()
    elif args.model == "Qwen2":
        model = Qwen2()
    elif args.model == "Mixtral":
        model = Mixtral()
    elif args.model == "Gemini":
        model = Gemini()
    elif args.model == "GPT":
        model = GPT()
    elif args.model == "Llama3.1":
        model = Llama3_1()
    elif args.model == "Llama3.2":
        model = Llama3_2()
    elif args.model == "Gemini1.5":
        model = Gemini1_5()
    elif args.model == "Phi3.5":
        model = Phi3_5()
    elif args.model == "MistralNemo":
        model = MistralNemo()
    elif args.model == "Gemini1.5_Flash":
        model = Gemini1_5_Flash()
    elif args.model == "Gemma2":
        model = Gemma2()
    elif args.model == "ChatGPT":
        model = ChatGPT()
    elif args.model == "Claude":
        model = Claude()
    elif args.model == "Phi3_4k":
        model = Phi3_4k()
    elif args.model == "Llama3.1-70B":
        model = Llama3_1_70B()
    elif args.model == "GPT4":
        model = GPT4()
    elif args.model == "GPT_4o_mini":
        model = GPT_4o_mini()
    elif args.model == "Llama3.2-90B":
        model = Llama3_2_90B()
    elif args.model == "All":
        for model in models:
            print(f"Evaluating model: {model.__class__.__name__}")
            evaluator = Evaluator(model=model, dataset=dataset)
            evaluator.evaluate()
            evaluator.print_results()
            print(model.__class__.__name__, "metrics")
            evaluator.compute_metrics()
            torch.cuda.empty_cache()
        return
    else:
        raise ValueError(f"Model {args.model} not recognized.")

    # Evaluate each model
    print(f"Evaluating model: {model.__class__.__name__}")
    evaluator = Evaluator(model=model, dataset=dataset, type=0)
    evaluator.evaluate()
    evaluator.print_results()
    print(model.__class__.__name__, "metrics")
    evaluator.compute_metrics()


if __name__ == "__main__":
    main()
