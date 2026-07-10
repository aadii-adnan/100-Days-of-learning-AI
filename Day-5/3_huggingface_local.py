from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
import os

os.environ['HF_HOME']='E/Hugging_face'

load_dotenv()  # Load environment variables from .env file

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    model_kwargs=dict(
        temperature=0.7,
        max_new_tokens=100,
    )
)



model = ChatHuggingFace(llm)

result = model.invoke("What is the capital of France?")

print(result.content)