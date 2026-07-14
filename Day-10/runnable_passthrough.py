from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate,load_prompt
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.7) 

prompt = PromptTemplate(
    template="Write a joke about {topic}",
    input_values= ['topic']

)

prompt2 = PromptTemplate(
    template="Explain the following {joke}",
    input_values= ['joke']
)


parser = StrOutputParser()

joke_gen = RunnableSequence(prompt, model, parser)

Parallel = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(prompt2, model, parser)
})

final_chain = RunnableSequence(joke_gen, Parallel)
result = final_chain.invoke({"topic": "Cricket"})
print (result['joke'])
