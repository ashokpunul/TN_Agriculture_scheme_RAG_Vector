
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    print("SUCCESS: OpenAI API key loaded.")
else:
    print("ERROR: OpenAI API key not found.")
    print("Check the .env filename and project folder.")