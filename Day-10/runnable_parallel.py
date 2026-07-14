from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate,load_prompt
from langchain_core.runnables import RunnableSequence, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.7) 

prompt = PromptTemplate(
    template="Write a linkedln post about {topic}",
    input_values= ['topic']

)

prompt2 = PromptTemplate(
    template="Write a tweet about the following {topic}",
    input_values= ['topic'] 
)  

parallel_chain = RunnableParallel({
    "linkedin" : RunnableSequence(prompt, model, StrOutputParser()),
    "tweet" : RunnableSequence(prompt2, model, StrOutputParser())
})

result = parallel_chain.invoke({"topic": "Cricket"})
print(result)