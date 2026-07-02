from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import  ResponseSchema, StructuredOutputParser

load_dotenv() 


 # Load environment variables from .env file

llm = HuggingFaceEndpoint(
      repo_id="CohereLabs/tiny-aya-global",
      task ="text-generation"
)

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name="name", description="The name of the character"),
    ResponseSchema(name="age", description="The age of the character"),
    ResponseSchema(name="city", description="The city of the character")
]

parser = StructuredOutputParser.from_response_schemas(schema)

template1 = PromptTemplate(
    template = """ give the name, age and city of fictional character {character} {format_instructions}""",
    input_variables=["character"],
    partial_variables={"format_instructions": parser.get_format_instructions()}


)

prompt1 = template1.invoke({"character": "Harry Potter"})

result = model.invoke(prompt1)
final_result = parser.parse(result.content)

print(final_result)