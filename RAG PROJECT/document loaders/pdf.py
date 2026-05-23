# # we are going to use document loaders such as, pdf (pypdf) and text (built in) to load documents and then use them for question answering


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter  # text splitter that splits the text into tokens, which are the basic units of text that a language model can understand. This is useful for splitting the text into smaller chunks that can be processed by the model, especially when dealing with large documents that may exceed the model's input limits.
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

# text_splitter = TokenTextSplitter(chunk_size=10, chunk_overlap=0)




print(Path(__file__).parent)
file_path = Path(__file__).parent / "notes.pdf"
data = PyPDFLoader(str(file_path))
docs = data.load()
print(len(docs)) # any kind of document contains metadata and page content, so we can access the page content using the page_content attribute of the document. This will give us the actual text content of the document, which we can then use for question answering or other tasks.
# as this is PDF file, there will be multiple documents(each page will be a document) in the list, so we can access the first page using docs[0].page_content
print(docs[0].page_content)



# splitter = TokenTextSplitter(chunk_size=900, chunk_overlap=10)
splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=10)

chunks = splitter.split_documents(docs);

print(len(chunks))
print(chunks[0].page_content)