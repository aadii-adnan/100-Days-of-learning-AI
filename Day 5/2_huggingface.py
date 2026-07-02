from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st

load_dotenv() 

st.header("Hugging Face Chat") 
user_input = st.text_input("Enter your message:") 
 # Load environment variables from .env file

llm = HuggingFaceEndpoint(
      repo_id="CohereLabs/tiny-aya-global",
      task ="text-generation"
)

model = ChatHuggingFace(llm=llm)

if st.button("Summarize"):
    result = model.invoke(user_input)
    st.write(result.content)


