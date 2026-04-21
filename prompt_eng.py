import pandas as pd
import torch
from transformers import pipeline
import re
import numpy as np
from sklearn.metrics import mean_absolute_error
import json

pipe = pipeline(
    "text-generation",
    model="google/gemma-3-4b-it",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    
)

print("Loaded Model--")

# Load the provided datasets
df = pd.read_csv("combined_scores.csv", index_col=None)

split_index = int(len(df) * 0.90)  #
train_df = df.iloc[:split_index]   
test_df = df.iloc[split_index:]
print(train_df.shape)
print(test_df.shape)


# Prepare few-shot examples from training data (selecting 5 random samples)
few_shot_examples = ""
for _, row in train_df.sample(n=50, random_state=42).iterrows():
    few_shot_examples += f"Input: Title: {row['title']}, Category ID: {row['category_id']}, Category: {row['category']}\n"
    few_shot_examples += f"Output: Score: {row['Score']}\n\n"


def extract_score(response_text):
    try:
        # Use regex to extract the JSON substring (including newlines)
        json_match = re.search(r'(\{.*\})', response_text, re.DOTALL)
        if not json_match:
            return None
        json_str = json_match.group(1)

        # Parse JSON
        response_json = json.loads(json_str)

        # Try accessing "result" key first
        if "result" in response_json and "Score" in response_json["result"]:
            score = response_json["result"]["Score"]
        elif "Score" in response_json:  # Check if "Score" exists at the root level
            score = response_json["Score"]
        else:
            raise KeyError("Score not found in expected locations")

        print(score)
        return score

    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error extracting score: {e}")
        return None

# Function to predict scores using LLaMA 3.2
def predict_score(test_instance):
    system_message = (
        "You are a model that predicts The Warwick-Edinburgh Mental Wellbeing Scales (WEMWBS) score. "
        "This score is always between 14 and 70. The input data consists of sensor readings from mobile phone sensors "
        "and social media data from YouTube, Instagram, and TikTok. Based on this input data, you have to predict the WEMWBS score."
    )

    # query = f"Here are some examples:\n\n{few_shot_examples}"
    query = f"Now, predict the Score for the following input:\n"
    query += f"Title: {test_instance['title']}, Category ID: {test_instance['category_id']}, Category: {test_instance['category']}\n\n"
    query += (
        "Provide a short reasoning first, then explicitly state the final predicted score at the end in JSON format. Make sure to return ONLY an instance of the JSON, NOT the schema itself. Do not add any additional information.\n"
        "JSON format: \n"
        "{\n"
        '    "result": {\n'
        '        "reason": "xxxxx",\n'
        '        "Score": x\n'
        "    }\n"
        "}\n"
        "Do not include any extra text."
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": query},
    ]

    outputs = pipe(
        messages,
        max_new_tokens=500,
        )
    response_text = outputs[0]["generated_text"][-1]["content"]
    print('response', response_text)
    return extract_score(response_text)  
    

def get_gpu_memory():
    """Returns GPU memory usage."""
    allocated = torch.cuda.memory_allocated() / 1024**2  # Convert to MB
    reserved = torch.cuda.memory_reserved() / 1024**2  # Convert to MB
    return allocated, reserved

# Before inference
torch.cuda.empty_cache()
start_allocated, start_reserved = get_gpu_memory()

# Apply the model on the test dataset
test_df["Predicted_Score"] = test_df.apply(predict_score, axis=1)

# After inference
end_allocated, end_reserved = get_gpu_memory()
print(f"Total GPU Memory Used During Inference: {end_allocated - start_allocated:.2f} MB")

# Convert to numeric and keep NaNs
test_df['Predicted_Score'] = pd.to_numeric(test_df['Predicted_Score'], errors='coerce')

# Save the results
test_df.to_csv("predicted_scores.csv", index=False)
print("Predictions saved to 'predicted_scores.csv'")

nan_count = test_df['Predicted_Score'].isna().sum()
print('Number of nan values in Predicted score', nan_count)

avg_score = test_df['Predicted_Score'].mean()
test_df ['Predicted_Score'] = test_df ['Predicted_Score'].fillna(avg_score)

mae = mean_absolute_error(test_df["Score"], test_df["Predicted_Score"])
print(f"Mean Absolute Error (MAE): {mae}")