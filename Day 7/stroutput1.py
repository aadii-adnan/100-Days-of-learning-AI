from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv() 


 # Load environment variables from .env file

llm = HuggingFaceEndpoint(
      repo_id="CohereLabs/tiny-aya-global",
      task ="text-generation"
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template = """ Write a detailed report on {topic} """,
    input_variables = ["topic"]
)


template2 = PromptTemplate(
    template = """ Write a 5 line summary on the text {text} """,
    input_variables = ["text"]
)

parser = StrOutputParser()


chain  = template1 | model | parser | template2 | model | parser

print (chain.invoke({"topic": "Climate Change"}))



