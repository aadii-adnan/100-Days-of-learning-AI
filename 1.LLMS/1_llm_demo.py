from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

llm = OpenAI(model_name="gpt-5.4-mini", temperature=0.7)

result = llm.invoke("What is the capital of France?")
print(result)