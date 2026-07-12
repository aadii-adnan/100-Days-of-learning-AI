from langchain_community.document_loaders import PyPDFLoader,WebBaseLoader
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate,load_prompt
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
import os

print(os.getenv("USER_AGENT"))

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.7) 

parser = StrOutputParser()


prompt = PromptTemplate(
    template = """ give ans of {question} from the {text}""",
    input_variables=["question","text"]
)


loader = WebBaseLoader("https://www.geeksforgeeks.org/python-programming-language/")

doc = loader.load()

chain = prompt | model | parser

result = chain.invoke({"question": "What is Python programming language?","text": doc[0].page_content})


