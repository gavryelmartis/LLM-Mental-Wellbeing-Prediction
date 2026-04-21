import pandas as pd
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error
import re
from datasets import Dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth.chat_templates import train_on_responses_only
import wandb
from unsloth.chat_templates import get_chat_template
from unsloth.chat_templates import get_chat_template, standardize_sharegpt
from datasets import Dataset, load_dataset
from transformers import BitsAndBytesConfig
import sys
import subprocess

def get_gpu_memory_nvidia_smi(index=1):
    """Returns the memory used by the GPU with the specified physical index as seen by nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '-i', str(index), '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True
        )
        used_memory = result.stdout.strip()
        print(f"GPU {index}: {used_memory} MiB used")
        return used_memory
    except Exception as e:
        print(f"Error retrieving GPU memory usage: {e}")
        return None


# ------------------------
# 1. Load Data and Setup
# ------------------------
df = pd.read_csv("combined_scores.csv", index_col=None)
df = df.dropna(subset=["category_id", "category"])  # Drop rows with missing values
df["category_id"] = df["category_id"].astype(int)  # Convert category_id to integer

split_index = int(len(df) * 0.90)
train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)


# ------------------------
# 2. Load and Prepare the Model for Fine-Tuning
# ------------------------
def train():
    try:
        run = wandb.init()
        run_name = "Llama3.2-1b"
        run.name = run_name
        run.save()

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        print("GPU Memory Before Model Load:")
        used_gpu_memory_1= get_gpu_memory_nvidia_smi(1)

        config = wandb.config
        wandb.log({
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "num_train_epochs": config.num_train_epochs,
            "max_seq_length": config.max_seq_length,
            "load_in": config.load_in,
            "GPU Memory Before Model Load":used_gpu_memory_1
        })

        # Determine loading method
        load_in_4bit = wandb.config.load_in == "4bit"
        load_in_8bit = wandb.config.load_in == "8bit"
        load_in_full = wandb.config.load_in == "full"
        dtype = None if wandb.config.load_in == "full" else "auto"

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,  # Set to True if using 4-bit quantization
            load_in_8bit=load_in_8bit,  # Set to True if using 8-bit quantization
            full_finetuning=load_in_full,
        )
        # Load the base model and tokenizer
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Llama-3.2-1B-Instruct", #"/user/home/gm17284/work/SAVED_MODELS/Llama-3.2-3B",
            max_seq_length=wandb.config.max_seq_length,
            dtype=dtype,
            quantization_config=quantization_config,
        )

        torch.cuda.synchronize()
        print("GPU Memory After Model Load:")
        used_gpu_memory_2 = get_gpu_memory_nvidia_smi(1)

        wandb.log({"GPU Memory After Model Load": used_gpu_memory_2})

    # Prepare the model for PEFT using LoRA
        model = FastLanguageModel.get_peft_model(
            model,
            r= wandb.config.lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha= wandb.config.lora_alpha,
            lora_dropout= wandb.config.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
        print('--------------------------2',next(model.parameters()).dtype)

        # ------------------------
        # 3. Prepare Fine-tuning Dataset
        # ------------------------

        # Load tokenizer and apply chat template
        tokenizer = get_chat_template(
            tokenizer,
            chat_template="llama-3.2", #gemma-3
        )

        # Define the formatting function for fine-tuning
        def format_instance(row):
            convos = [
                {
                    "role": "system",
                    "content": (
                        "You are a model that predicts The Warwick-Edinburgh Mental Wellbeing Scales (WEMWBS) score. "
                        "This score is always between 14 and 70. The input data consists of sensor readings from mobile phone sensors "
                        "and social media data from YouTube, Instagram, and TikTok."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Predict the WEMWBS score given the input data.\n\n"
                        f"Title: {row['title']}, Category ID: {row['category_id']}, Category: {row['category']}"
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        '{{\n'
                        '    "result": {\n'
                        f'        "Score": {row["Score"]}\n'
                        '    }\n'
                        '}}'
                    ),
                },
            ]
            return {"conversations": convos}

        # Convert the train DataFrame into the appropriate format
        train_dataset = Dataset.from_pandas(train_df)
        train_dataset = train_dataset.map(format_instance)

        def formatting_prompts_func(examples):
            convos = examples["conversations"]
            texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
            return { "text" : texts, }
        pass


        train_dataset = standardize_sharegpt(train_dataset)
        train_dataset = train_dataset.map(formatting_prompts_func, batched = True,)


        # ------------------------
        # 4. Fine-tune the Model
        # ------------------------
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            dataset_text_field="text",
            max_seq_length=wandb.config.max_seq_length,
            dataset_num_proc=2,
            packing=False,  # Change if your sequences are short and can be packed.
            args=TrainingArguments(
                per_device_train_batch_size=wandb.config.batch_size,
                gradient_accumulation_steps=4,
                num_train_epochs=wandb.config.num_train_epochs,
                learning_rate=wandb.config.learning_rate,
                fp16=fp16,
                bf16=bf16,
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir="outputs",
                report_to="wandb",
            ),
        )


        print("Starting fine-tuning...")
        trainer.train()
        print("Fine-tuning complete.")

        torch.cuda.synchronize()
        print("GPU Memory After Model Training:")
        used_gpu_memory_4 = get_gpu_memory_nvidia_smi(1)
        wandb.log({"GPU Memory After Model Training": used_gpu_memory_4})

        def extract_score(response_text):
            """
            Extracts the numerical score from the response text by searching for all occurrences
            of the pattern '"result": { ... "Score": <number>' and then returning the number from
            the last occurrence. This ensures that we ignore the sample JSON (with placeholder 'X')
            and only return the actual predicted score.
            """
            try:
                # Find all matches for the score number in a block containing "result" and "Score".
                matches = re.findall(r'"result"\s*:\s*\{[^}]*"Score"\s*:\s*([0-9]+)', response_text, re.DOTALL)
                if matches:

                    score = int(matches[-1])
                    print("Extracted Score:", score)
                    return score
                else:
                    print("No valid score found.")
                    return None
            except Exception as e:
                print("Error extracting score:", e)
                return None 
    

        def predict_score(test_instance):

            messages = [
                {"role": "system", "content": (
                    "You are a model that predicts the WEMWBS score from input data consisting of sensor readings from mobile phone sensors and social media data. "
                    "The score is always between 14 and 70. Predict the score based on the given title, category ID, and category."
                )},
                {"role": "user", "content": (
                    f"Predict the WEMWBS score given the YouTube data.\n\n"
                    f"Title: {test_instance['title']}, Category ID: {test_instance['category_id']}, Category: {test_instance['category']}\n\n"
                    "Provide your reasoning first, then explicitly state the final predicted score in JSON format:\n"
                    '{{\n'
                    '    "result": {\n'
                    '        "reason": "Your reasoning here",\n'
                    '        "Score": X\n'
                    '    }\n'
                    '}}\n'
                    "Do not include any extra text."
                )}
            ]

            # Tokenize input message
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True, 
                return_tensors="pt",
            ).to("cuda")

            # Generate response
            outputs = model.generate(
                input_ids=inputs,
                max_new_tokens=1024,
                use_cache=True,
                temperature=1.5,
                min_p=0.1
            )

            # Decode response
            response_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
            print("-----Model Response:--------------")
            print("Model Response:", response_text)

            # Extract and return the predicted score
            return extract_score(response_text)

        tokenizer = get_chat_template(tokenizer, chat_template="llama-3.2")

        # Enable native 2x faster inference for the model.
        FastLanguageModel.for_inference(model)

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        test_df["Predicted_Score"] = test_df.apply(predict_score, axis=1)

        torch.cuda.synchronize()
        print("GPU Memory After Inference:")
        used_gpu_memory_3 = get_gpu_memory_nvidia_smi(1)

        wandb.log({"GPU Memory After Inference": used_gpu_memory_3})

        # Handle any missing predictions by filling with the average predicted score.
        nan_count = test_df['Predicted_Score'].isna().sum()
        print('Number of nan values in Predicted score', nan_count)

        avg_score = test_df['Predicted_Score'].mean()
        test_df['Predicted_Score'] = test_df['Predicted_Score'].fillna(avg_score)

        # Compute Mean Absolute Error (MAE)
        mae = mean_absolute_error(test_df["Score"], test_df["Predicted_Score"])
        print(f"Mean Absolute Error (MAE): {mae}")

        wandb.log({
            "Mean Absolute Error": mae,
            "Predicted nan values": nan_count,
            "Predicted scores": test_df['Predicted_Score'].tolist()
        })


        for i in range(torch.cuda.device_count()):
            print(f"Memory summary for device {i}:")
            print(torch.cuda.memory_summary(torch.device(f'cuda:{i}'), abbreviated=False))

        wandb.finish()

    except Exception as e:
        print("\n Error occurred during this sweep run:")
        print(e)
        print("Exiting to stop all further sweep runs.")
        wandb.finish()  
        sys.exit(1)


sweep_config = {
    "method": "bayes", 
    "metric": {"name": "loss", "goal": "minimize"},
    "parameters": {
        "lora_r": {"values": [8, 16, 32]},
        "lora_alpha": {"values": [8, 16, 32]},
        "lora_dropout": {"values": [0, 0.1, 0.2]},
        "learning_rate":  {"distribution": "log_uniform_values", "min": 0.00005, "max": 0.0001},
        "batch_size": {"values": [2, 4, 8]},
        "num_train_epochs": {"values": [1, 2, 3, 5]},
        "max_seq_length": {"values": [512, 1024, 2048]},
        "load_in": {"values": ["4bit","8bit","full"]},
    },
}

# Run the sweep
sweep_id = wandb.sweep(sweep_config, project="llama3.2-1bInstruct-finetune")
wandb.agent(sweep_id=sweep_id, function=train, count=500)










