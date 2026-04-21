# WEMWBS Score Predictor using Large Language Models
This repository contains code to predict the Warwick-Edinburgh Mental Wellbeing Scales (WEMWBS) score (ranging from 14 to 70) based on social media data (YouTube, Instagram, TikTok) and mobile phone sensor readings.

To tackle this task, this project explores and compares three distinct Large Language Model (LLM) paradigms:
1. Prompt Engineering (Zero-shot and few-shot) (`prompt_eng.py`)
2. Parameter-Efficient Fine-Tuning (PEFT/LoRA) (`finetune_llm.py`)
3. Retrieval-Augmented Generation (RAG) (`rag.py`)

## Dataset Requirement & Privacy Notice
Data Privacy Notice: Due to the sensitive and personal nature of the collected data (mobile phone sensor readings and personal social media usage), the dataset cannot be uploaded or publicly shared in this repository.
To run this code on your own data, you will need to provide your own dataset in the root directory and change the column names accordingly. 
Note: The scripts automatically perform a 90/10 Train/Test split on the provided CSV file.

## Setup and Installation
1. General Prerequisites
You will need Python 3.8+ and a CUDA-compatible GPU (required for `finetune_llm.py`).

2. Specific Setup for Fine-Tuning (`finetune_llm.py`)
The fine-tuning script uses Unsloth for 2x faster, memory-efficient training.
Please follow the official Unsloth installation guide for your specific CUDA version.
You will also need a Weights & Biases (W&B) account for hyperparameter tuning.

4. Specific Setup for RAG (rag.py)
The RAG script uses LangChain and FAISS.

## How to Run
1. Prompt Engineering
This script loads `Gemma-3-4b-it` and evaluates the test set using prompt instructions to enforce JSON output.
`python prompt_eng.py`
Outputs: Extracts the predicted scores, calculates the Mean Absolute Error (MAE), and saves results to `predicted_scores.csv`

2. Fine-Tuning (W&B Sweep)
This script initialises a Bayesian sweep via W&B to find the optimal LoRA hyperparameters (rank, alpha, learning rate, batch size, quantization).
`python finetune_llm.py`
Outputs: Logs training metrics and memory usage to your W&B dashboard, evaluates the test set, and calculates MAE.

3. Retrieval-Augmented Generation (RAG)
This script embeds the training dataset into a FAISS vector database. For every test instance, it retrieves the top 5 most similar examples to provide context to the model. It utilises a ThreadPoolExecutor for concurrent predictions.
`python rag.py`
Outputs: Calculates MAE and saves the results to `predicted_scores_rag.csv`

## Evaluation
The primary evaluation metric for this regression task is Mean Absolute Error (MAE).
All scripts feature built-in robust JSON/Regex parsers to extract the numerical score from the LLM's text output. If the model fails to format its output correctly, the scripts gracefully handle NaN values by imputing the average predicted score before calculating the final MAE.

