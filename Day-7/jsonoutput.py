from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv() 


 # Load environment variables from .env file

llm = HuggingFaceEndpoint(
      repo_id="CohereLabs/tiny-aya-global",
      task ="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template1 = PromptTemplate(
    template = """ give the name, age and city of fictional character {format_instructions} """,
    input_variables=[],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

prompt = template1.invoke({})

result = model.invoke(prompt)

final_result = parser.parse(result.content)


print(final_result)
