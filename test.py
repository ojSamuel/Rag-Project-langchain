from dotenv import load_dotenv
import os

# Force-read directly from the file, bypassing os.environ cache
load_dotenv(dotenv_path=".env", override=True)

# Double-check by reading the file directly
from dotenv import dotenv_values
config = dotenv_values(".env")
print("From file directly:", config.get("GOOGLE_API_KEY"))
print("From os.getenv:", os.getenv("GOOGLE_API_KEY"))
with open(".env") as f:
       print(f.read())