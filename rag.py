import pandas as pd
import numpy as np
import re
import json
from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sklearn.metrics import mean_absolute_error
from concurrent.futures import ThreadPoolExecutor
import psutil
import os

# Load dataset
df = pd.read_csv("combined_scores.csv", index_col=None)

# Splitting train and test data
split_index = int(len(df) * 0.90)
train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

train_df["text"] = train_df.apply(lambda row: f"Title: {row['title']}, Category ID: {row['category_id']}, Category: {row['category']}, Score: {row['Score']}", axis=1)

# Convert DataFrame to LangChain Documents
loader = DataFrameLoader(train_df, page_content_column="text")
docs = loader.load()

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250, chunk_overlap=0)
doc_splits = text_splitter.split_documents(docs)

# Load SentenceTransformer embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create FAISS vector database
vector_db = FAISS.from_documents(doc_splits, embedding_model)
retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Define system message to instruct the model
system_message = (
    "You are a model that predicts The Warwick-Edinburgh Mental Wellbeing Scales (WEMWBS) score. "
    "This score is always between 14 and 70. The input data consists of sensor readings from mobile phone sensors "
    "and social media data from YouTube, Instagram, and TikTok. Based on this input data, you have to predict the WEMWBS score."
)

# Define prompt template for RAG
prompt = PromptTemplate(
    template="""Use the following documents to answer the question:

{documents}

Now, predict the Score for the following input:
Title: {title}, Category ID: {category_id}, Category: {category}

Provide your reasoning first, then explicitly state the final predicted score at the end in JSON format. Make sure to return ONLY an instance of the JSON, NOT the schema itself. Do not add any additional information.

JSON format:
{{
    "result": {{
        "reason": "xxxxx",
        "Score": x
    }}
}}
Do not include any extra text.""",
    input_variables=["title", "category_id", "category", "documents"],
)


# Initialize the LLM
llm = ChatOllama(model="llama3.2:1b", temperature=0, system_message=system_message)
rag_chain = prompt | llm | StrOutputParser()

# Define the RAG application class
class RAGScorePredictor:
    def __init__(self, retriever, rag_chain):
        self.retriever = retriever
        self.rag_chain = rag_chain
    
    def predict(self, test_instance):
        query_text = f"Title: {test_instance['title']}, Category ID: {test_instance['category_id']}, Category: {test_instance['category']}"
        retrieved_docs = self.retriever.invoke(query_text)
        print('retrieved_docs', retrieved_docs)
        doc_texts = "\n".join([doc.page_content for doc in retrieved_docs])
        print('doc_texts', doc_texts)
        response = self.rag_chain.invoke({"title": test_instance['title'], "category_id": test_instance['category_id'], "category": test_instance['category'], "documents": doc_texts})
        print(response)

        return self.extract_score(response)
    
    @staticmethod
    def extract_score(response_text):
        try:
            # Use regex to extract JSON substring (including newlines)
            json_match = re.search(r'(\{.*\})', response_text, re.DOTALL)
            if not json_match:
                print("No valid JSON found in response.")
                return None
            
            json_str = json_match.group(1)

            # Attempt to fix common formatting errors
            json_str = json_str.strip()
            
            # Ensure closing brackets are correctly balanced
            open_braces = json_str.count("{")
            close_braces = json_str.count("}")
            if open_braces > close_braces:
                json_str += "}" * (open_braces - close_braces)  # Add missing closing braces

            # Parse JSON
            response_json = json.loads(json_str)

            # Extract score from valid keys
            if "result" in response_json and isinstance(response_json["result"], dict):
                score = response_json["result"].get("Score")
            elif "Score" in response_json:
                score = response_json["Score"]
            else:
                raise KeyError("Score not found in expected locations")

            # Ensure score is a valid number
            if not isinstance(score, (int, float)):
                raise ValueError(f"Extracted score is not a number: {score}")

            return score

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error extracting score: {e}\nResponse Text: {response_text}")
            return None


# Initialize the RAG application
rag_predictor = RAGScorePredictor(retriever, rag_chain)

# Apply the RAG pipeline to predict scores for the test dataset
def parallel_predict(test_instance):
    return rag_predictor.predict(test_instance)

# with ThreadPoolExecutor(max_workers=4) as executor:
with ThreadPoolExecutor(max_workers=5) as executor:
    test_df["Predicted_Score"] = list(executor.map(parallel_predict, [row for _, row in test_df.iterrows()]))

# Convert to numeric and keep NaNs
test_df['Predicted_Score'] = pd.to_numeric(test_df['Predicted_Score'], errors='coerce')

# Save the results
test_df.to_csv("predicted_scores_rag.csv", index=False)
print("Predictions saved to 'predicted_scores_rag.csv'")

nan_count = test_df['Predicted_Score'].isna().sum()
print('Number of nan values in Predicted score', nan_count)

avg_score = test_df['Predicted_Score'].mean()
test_df['Predicted_Score'] = test_df['Predicted_Score'].fillna(avg_score)

mae = mean_absolute_error(test_df["Score"], test_df["Predicted_Score"])
print(f"Mean Absolute Error (MAE): {mae}")
