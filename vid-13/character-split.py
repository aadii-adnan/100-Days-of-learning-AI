from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("dl-curriculum.pdf")

doc =loader.load()



# text = "This is a sample text that we will split into smaller chunks based on character count. The CharacterTextSplitter allows us to specify the maximum number of characters per chunk, and it will handle the splitting for us."

text_splitter = CharacterTextSplitter(
    chunk_size=50,  # Maximum number of characters per chunk
    chunk_overlap=0, # Number of characters to overlap between chunks
    separator=''
)
chunks = text_splitter.split_documents(doc)

print (chunks[0].page_content)
