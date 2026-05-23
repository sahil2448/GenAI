# # we are going to use document loaders such as, pdf (pypdf) and text (built in) to load documents and then use them for question answering


from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter 
from pathlib import Path


splitter = CharacterTextSplitter(
    separator="", # empty string means we will split the text into individual characters, 
    chunk_size=10,
    chunk_overlap=1,
)


print(Path(__file__).parent) # this will give us the path of the current file, which is text.py, and then we can use it to construct the path to the notes.txt file, which is in the same directory as text.py. This way we can load the notes.txt file without hardcoding the path, making our code more portable and easier to run on different machines.
file_path = Path(__file__).parent / "notes.txt"
data = TextLoader(str(file_path))
docs = data.load()
print(docs) # any kind of document contains metadata and page content, so we can access the page content using the page_content attribute of the document. This will give us the actual text content of the document, which we can then use for question answering or other tasks.
# if it's a text file..there will be only one document in the list, so we can access it using docs[0].page_content

chunks = splitter.split_documents(docs)
print(len(chunks))

for chunk in chunks:
    print(chunk.page_content)
    print()
    print()


### use read the documentation for the character text splitter

## from Doc -- Character-based splitting is the simplest approach to text splitting. It divides text using a specified character sequence (default: "\n\n"), with chunk length measured by the number of characters...

# if i don't provide separator then it will split the text into individual words, meaning that the chunk will be a word and the chunk size will be the number of characters in that word, which is not what we want. So we need to provide an empty string as the separator to split the text into individual characters, and then we can specify the chunk size and chunk overlap as needed. 

# if we provide a separator, then it will split the text based on that separator, and the chunk size will be the number of characters in that chunk, which is not what we want. So we need to provide an empty string as the separator to split the text into individual characters, and then we can specify the chunk size and chunk overlap as needed.