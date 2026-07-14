from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate,load_prompt
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.7) 

prmpt = PromptTemplate(
    template="Write a report about {topic}",
    input_values= ['topic']

)

prompt1 = PromptTemplate(
    template="Summerize the following {report}",
    input_values= ['report']
)

parser = StrOutputParser()

Report_gen = RunnableSequence(prmpt, model, parser)

branch_chain = RunnableBranch(
   (RunnableLambda(lambda x: len(x.split()) > 500), RunnableSequence(prompt1, model, parser)),
   RunnablePassthrough()
)

final_chain = RunnableSequence(Report_gen, branch_chain)
print(final_chain.invoke({"topic": "Cricket"}))