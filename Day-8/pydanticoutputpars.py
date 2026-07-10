from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field



load_dotenv() 


 # Load environment variables from .env file

llm = HuggingFaceEndpoint(
      repo_id="CohereLabs/tiny-aya-global",
      task ="text-generation"
)

model = ChatHuggingFace(llm=llm)

class person (BaseModel):
    name: str = Field(description="The name of the character")
    age: int = Field(gt=18, description="The age of the character")
    city: str = Field (description="The city of the character")


parser = PydanticOutputParser(pydantic_object=person)

template1 = PromptTemplate(
    template = """ give the name, age and city of fictional character that belong to a {place} /n 
{format_instructions}""",
    input_variables=["place"], 
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = template1 | model | parser



result = chain.invoke({"place": "fantasy world"})
print (result)
