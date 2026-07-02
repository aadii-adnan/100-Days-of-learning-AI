from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal






load_dotenv() 



 # Load environment variables from .env file

class feedback(BaseModel):
    sentiment : Literal["positive", "negative"] = Field(description="The sentiment of the feedback")    
 

parser = StrOutputParser()
parser2 = PydanticOutputParser(pydantic_object=feedback)

llm = HuggingFaceEndpoint(
      repo_id="CohereLabs/tiny-aya-global",
      task ="text-generation"
)

model1 = ChatHuggingFace(llm=llm)


model2 = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.7)    



prompt1 = PromptTemplate(
    template = """ Analyze the sentiment of the following text: {text} /n {format_instructions}""",
    input_variables=["text"],
    partial_variables={"format_instructions": parser2.get_format_instructions()}
)


prompt2 = PromptTemplate(
    template = """ give an appropriate response to the positive feedback: {text}""",
    input_variables=["text"]
)


prompt3 = PromptTemplate(
    template = """ give an appropriate response to the negative feedback: {text}""",
    input_variables=["text"]
)
analysis = prompt1 | model1 | parser2

branch = RunnableBranch(
    (lambda x: x.sentiment == "positive", prompt2 | model1 | parser),
    (lambda x: x.sentiment == "negative", prompt3 | model2 | parser),
    RunnableLambda(lambda x: "Invalid sentiment")
)

chain = analysis | branch

result = chain.invoke({ "text" : "I love the new features in your product! It's so user-friendly and intuitive." })

print (result)
