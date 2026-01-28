import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Access the variable
password = os.getenv("GOOGLE_API_KEY")

print(f"this is the password {password}")