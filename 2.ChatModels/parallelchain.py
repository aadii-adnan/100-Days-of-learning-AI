from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel




load_dotenv() 


 # Load environment variables from .env file

llm = HuggingFaceEndpoint(
      repo_id="CohereLabs/tiny-aya-global",
      task ="text-generation"
)

model1 = ChatHuggingFace(llm=llm)


model2 = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.7)    



prompt1 = PromptTemplate(
    template = """ give a summary of the text {text}""",
    input_variables=["text"]
)

prompt2 = PromptTemplate(
    template = """ give a quiz based on the {text}""",
    input_variables=["text"]
)

prompt3 = PromptTemplate(
    template = """ marge the summary and quiz of the text {summary} and {quiz} """,
    input_variables=["summary", "quiz"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {
        "summary" : prompt1 | model1 | parser,
        "quiz" : prompt2 | model2 | parser
    }

)

chain = parallel_chain | prompt3 | model1 | parser

result = chain.invoke({ "text" : "Reading is a transformative habit that broadens our perspectives and sharpens cognitive abilities. By immersing ourselves in diverse stories and complex ideas, we develop greater empathy and improve critical thinking. Ultimately, books serve as gateways to endless knowledge, allowing us to continuously learn and grow from the comfort of our surroundings." })

print (result)

chain.get_graph().print_ascii()

