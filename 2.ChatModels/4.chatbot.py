from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate,load_prompt
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.7) 

chat_template = ChatPromptTemplate([
    ('system', 'you are a {domain} expert'),
    ('human', 'Summarize the research paper {topic}')


])

prompt = chat_template.invoke({'domain': 'AI', 'topic': 'Attention Is All You Need'})

print (prompt)



# messages=[
#     SystemMessage(content="You are a helpful assistant."),
#     HumanMessage(content="Summarize the research paper 'whats the capital of pakistan?'")
# ]

# result = model.invoke(messages)

# messages.append(AIMessage(content = result.content[0]["text"]))

# print (messages)