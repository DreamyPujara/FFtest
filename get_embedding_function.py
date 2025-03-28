#from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "AIzaSyAHKUAiO-kLn0tuP9jYjyzfLZ0ri49JAOc"


load_dotenv()

def get_embedding_function():

    #embeddings = OllamaEmbeddings(model="llama3.2")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    return embeddings
 