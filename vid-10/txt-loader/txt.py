from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate,load_prompt
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.7) 

parser = StrOutputParser()


prompt = PromptTemplate(
    template = """ give summary of the {text}""",
    input_variables=["text"]
)



Loader = TextLoader("home.txt")

doc = Loader.load()

print (doc[0].page_content)

chain = prompt | model | parser

print (chain.invoke({"text": doc[0].page_content}))