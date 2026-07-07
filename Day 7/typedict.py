
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field, Optional


load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.7) 

json_schema = {
     "title" : "ResearchPaperSummary",
     "description" : "A summary of a research paper, including key themes, sentiment analysis, and pros and cons.",
     "type" : "object",
     "properties" : {
          "key_themes" : {
               "type" : "array",
               "items" : {"type" : "string"},
                "description" : "The key themes discussed in the research paper"
          }
     },

     "required" : ["key_themes", "summary", "Sentiment"]    
}

class Review (BaseModel):
     key_themes :  [list[str]] = Field(description="The key themes discussed in the research paper")
     summary: str = Field(description="A brief summary of the research paper")
     Sentiment: Literal['positive', 'negative', 'neutral'] = Field(description="The sentiment of the research paper, e.g., positive, negative, neutral")
     pros : Optional[list[str]] = Field(default = 'none', description="The positive aspects of the research paper")
     cons : Optional[list[str]] = Field(default = 'none', description="The negative aspects of the research paper")   



structured_model = model.with_structured_output(Review)

result = structured_model.invoke("Summarize the research paper 'Attention Is All You Need' and provide sentiment analysis.")

print (result)

