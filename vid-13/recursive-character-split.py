from langchain_text_splitters import CharacterTextSplitter,RecursiveCharacterTextSplitter


text = "This is a sample text that we will split into smaller chunks based on character count. The CharacterTextSplitter allows us to specify the maximum number of characters per chunk, and it will handle the splitting for us."

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,  # Maximum number of characters per chunk
    chunk_overlap=0, # Number of characters to overlap between chunks
)
chunks = text_splitter.split_text(text)


print (chunks[0])