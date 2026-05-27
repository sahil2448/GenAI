# load pdf
# split into chunks 
# create embeddings
# store into chroma db
from langchain_community.document_loaders import PyPDFLoader # pdf loader
from langchain_text_splitters import RecursiveCharacterTextSplitter # chunking
from langchain_openai import OpenAIEmbeddings # embedding model
from langchain_community.vectorstores import chroma # vector store
from dotenv import load_dotenv
load_dotenv()

data = PyPDFLoader("document loaders/notes.pdf")
docs = data.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=100)

chunks = splitter.split_documents(docs);

embedding_model = OpenAIEmbeddings()

vector_store = Chroma.from_documents( # we are passing chunks instead of docs because we want to create embeddings for the chunks, not the original documents. This is because the original documents may be too large to process and may exceed the input limits of the embedding model. By splitting the documents into smaller chunks, we can create embeddings for each chunk and store them in the vector store, which allows us to retrieve relevant chunks based on similarity search.
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma-db"
)