from fastapi import FastAPI
from pydantic import BaseModel
from keybert import KeyBERT
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import requests
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel
import subprocess
from transformers import pipeline
import PyPDF2
import docx
import pandas as pd
import os

# Load models
kw_model = KeyBERT()
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
command_suggester = pipeline("text-generation", model="gpt2")

app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow requests from your Next.js frontend
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

class UserInput(BaseModel):
    query: str
    type: str = "search"  # Can be "search", "cmd", or "find"
    base_path: str = None  # Optional: custom file path

class ExecuteCommandRequest(BaseModel):
    command: str

class SearchRequest(BaseModel):
    query: str
    base_path: str # = "./test_files"

def generate_keywords(query: str):
    """
    Extracts keywords using KeyBERT.
    """
    keywords = kw_model.extract_keywords(query, keyphrase_ngram_range=(1, 1), stop_words='english', top_n=5)
    return list(set([kw[0] for kw in keywords]))  # Remove duplicates

def prioritize_keywords(query: str, keywords: list):
    """
    Prioritizes keywords based on similarity to the query.
    """
    query_embedding = sentence_model.encode(query, convert_to_tensor=True)
    keyword_embeddings = sentence_model.encode(keywords, convert_to_tensor=True)
    similarities = cosine_similarity(query_embedding.reshape(1, -1), keyword_embeddings)[0]
    return [kw for _, kw in sorted(zip(similarities, keywords), reverse=True)]

def prioritize_results(query: str, results: list) -> list:
    """
    Prioritizes search results based on relevance to the query.
    """
    query_embedding = sentence_model.encode(query, convert_to_tensor=True)
    result_embeddings = sentence_model.encode([result["name"] for result in results], convert_to_tensor=True)
    similarities = cosine_similarity(query_embedding.reshape(1, -1), result_embeddings)[0]
    prioritized_results = [result for _, result in sorted(zip(similarities, results), reverse=True)]
    return prioritized_results[:4]  # Return top 4 results

def search_rust_backend(keywords: list, base_path: str):
    """
    Sends a search request to the Rust backend and returns the results.
    """
    url = "http://127.0.0.1:3030/search"
    payload = {"keywords": keywords, "base_path": base_path}
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        return response.json().get("results", [])
    else:
        return {"error": f"Rust backend error: {response.status_code} - {response.text}"}

@app.post("/search")
def search_files(request: SearchRequest):
    keywords = generate_keywords(request.query)
    prioritized_keywords = prioritize_keywords(request.query, keywords)
    search_results = search_rust_backend(prioritized_keywords, request.base_path)
    
    if isinstance(search_results, list):  # Check if search_results is a list
        prioritized_results = prioritize_results(request.query, search_results)
        return {"keywords": prioritized_keywords, "results": prioritized_results}
    else:
        return {"error": search_results.get("error", "Unknown error")}  # Handle error case

@app.post("/cmd")
def suggest_command(request: UserInput):
    """
    Suggests a command based on the user's input.
    """
    try:
        # Use the model to interpret the query
        suggested_command = command_suggester(request.query, max_length=10, num_return_sequences=1)[0]["generated_text"]
        return {"command": suggested_command.strip()}
    except Exception as e:
        return {"error": f"Failed to suggest command: {str(e)}"}

@app.post("/find")
def find_in_document(request: UserInput):
    """
    Searches for the requested information in files (ignoring folders) and returns lines containing the keywords.
    """
    try:
        # Check if file_path is provided; if not, use default base path
        path_to_search = request.base_path if request.base_path else "./test_files"
        
        # Generate keywords from the query
        keywords = generate_keywords(request.query)
        
        # Send a request to search the backend
        search_results = search_rust_backend(keywords, path_to_search)
        
        if isinstance(search_results, list):  # If we got results back as a list
            results = []
            for result in search_results:
                try:
                    # Check if the result is a file (not a directory)
                    if os.path.isfile(result["path"]):
                        # Convert the content of the file to text
                        text = convert_to_text(result["path"])
                        
                        # Find matches in the converted text
                        matches = find_matches(text, keywords)
                       
                        if matches:
                            # Append results if matches are found
                            results.append({"path": result["path"], "matches": matches, "icon": result["icon"]})
                except Exception as e:
                    print(f"Failed to scan {result['path']}: {str(e)}")
           
            return {"results": results}
        else:
            # Handle error when backend returns something unexpected
            return {"error": search_results.get("error", "Unknown error")}
    except Exception as e:
        return {"error": f"Failed to scan documents: {str(e)}"}
    

def convert_to_text(file_path: str) -> str:
    """
    Converts a file to plain text.
    """
    # Use os.path.abspath to ensure we have the absolute path
    file_path = os.path.abspath(file_path)

    if file_path.endswith(".pdf"):
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
            return text
    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)
    elif file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
        return df.to_string()
    elif file_path.endswith(".txt"):  # Support for .txt files
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read()
        except FileNotFoundError:
            raise ValueError(f"File not found: {file_path}")
    elif os.path.isdir(file_path):  # Handle folders
        results = []
        for root, dirs, files in os.walk(file_path):
            for file in files:
                full_path = os.path.join(root, file)
                if full_path.endswith(".txt"):
                    try:
                        with open(full_path, "r", encoding="utf-8") as f:
                            results.append(f.read())
                    except FileNotFoundError:
                        raise ValueError(f"File not found: {full_path}")
        print(results)
        return "\n".join(results)  # Join text from all files in the folder
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    
def find_matches(text: str, keywords: list) -> list:
    """
    Finds and returns lines in the text that match any keyword in the list.
    Each match is returned with the original line text.
    """
    matches = []
    for keyword in keywords:
        # Go through each line in the document and check if the keyword appears
        for line in text.splitlines():
            if keyword.lower() in line.lower():  # Case-insensitive matching
                matches.append(line)
    return matches

@app.post("/api/execute")
def execute_os_command(command: str):
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return {
            "success": True,
            "output": result.stdout,
            "error": result.stderr,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }