import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_huggingface import ChatHuggingFace
# from langchain_ollama import ChatOllama

load_dotenv()

# My Keys

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEI")
HUGGINGFACE_API_KEY = os.getenv("GROQ_API_KEI")

# My LLMs

llm = ChatGroq(
    model='meta-llama/llama-4-maverick-17b-128e-instruct',
    api_key=GROQ_API_KEY,
    temperature=.2
)

# llm = ChatGoogleGenerativeAI(
#         api_key=GEMINI_API_KEY,
#         model='gemini-2.0-flash'
# )

# llm = ChatGroq(
#     model='meta-llama/llama-4-scout-17b-16e-instruct',
#     api_key=GROQ_API_KEY,
#     temperature=.2
# )
