import json
import sys

# Load the inference results
def load_inference_results(file_path):
    with open(file_path, "r") as f:
        return json.load(f)
    
# Get the filename from the command line arguments
if len(sys.argv) < 2:
    print("Usage: python script.py <filename>")
    sys.exit(1)

filename = sys.argv[1]
inference_results = load_inference_results(filename)

# Function to calculate categorical accuracy and overall accuracy
def evaluate_accuracy(inference_results):
    correct_count = 0
    total_count = len(inference_results)
    
    # Dictionary to hold the count of correct answers for each category
    category_correct = {}
    category_total = {}
    
    # Loop through the inference results to calculate accuracy
    for result in inference_results:
        predicted = result['response']  # Predicted option number
        actual = result['correct_option']  # Correct answer option (0-indexed)
        category = result['classification'] if result['classification'] is not None else "unanswerable"
        category = category.replace("spatial_", "")
        
        if category == "unanswerable":
            print(f"Unanswerable question")
        
        if category not in category_correct:
            category_correct[category] = 0
            category_total[category] = 0
                
        # Check if the prediction is correct
        if str(predicted) == str(actual):
            correct_count += 1  
            category_correct[category] += 1
        
        # Increment the total count for the category
        category_total[category] = category_total.get(category, 0) + 1
    
    # Overall accuracy
    overall_accuracy = correct_count / total_count if total_count > 0 else 0
    
    # Category-wise accuracy
    category_accuracy = {}
    for category, correct in category_correct.items():
        total = category_total[category]
        category_accuracy[category] = correct / total if total > 0 else 0

    # Print categorical count
    print("Category-wise count:")
    for category, correct in category_correct.items():
        total = category_total[category]
        print(f"Category '{category}': {total} examples, {correct} correct")

    return overall_accuracy, category_accuracy

# Evaluate accuracy
overall_accuracy, category_accuracy = evaluate_accuracy(inference_results)


    
# Output the results
print(f"Overall Accuracy: {overall_accuracy:.4f}")
for category, accuracy in category_accuracy.items():
    print(f"Accuracy for category '{category}': {accuracy:.4f}")
