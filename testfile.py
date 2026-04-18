import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

emb = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",  # use full model path
    google_api_key=api_key,
    client_options={"api_endpoint": "generativelanguage.googleapis.com"},
)

vector = emb.embed_query("hello world")
print(len(vector))


#import os
#from dotenv import dotenv_values

# Read directly from file, bypassing os.environ
#config = dotenv_values(".env")
#print("Key from file:", config.get("GOOGLE_API_KEY"))
#print("Key from os.environ:", os.environ.get("GOOGLE_API_KEY"))
