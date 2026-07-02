from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="LiquidAI/LFM2.5-Embedding-350M",
                                  
                                   
                                   )

documents = [
    "Shahid Afridi is an Pakistani cricketer known for his aggressive batting style.",
    "Wasim Akram is a world-class Pakistani fast bowler.",
    "The capital of Pakistan is Islamabad.",
    "Paris is known as the city of lights.",
    "Machine learning models require training data."
]

doc = embeddings.embed_documents(documents)

query = "Who is the best Pakistani cricketer?"
query_embedding = embeddings.embed_query(query)

query_embedding_reshaped = np.array(query_embedding).reshape(1, -1)

scores = cosine_similarity(query_embedding_reshaped, doc)

index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

print(query)
print(f"Most relevant document: {documents[index]}")
print (f"Cosine similarity score: {score}")
