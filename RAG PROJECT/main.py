#  code for text document loader

# from dotenv import load_dotenv
# from langchain_mistralai import ChatMistralAI
# from langchain_community.document_loaders import TextLoader
# from langchain_core.prompts import ChatPromptTemplate
# from pathlib import Path

# load_dotenv()


# # file_path = Path(__file__).parent /"docuement loader/" "notes.txt"
# # data = TextLoader(str(file_path))
# data = TextLoader("document loaders/notes.txt")
# docs = data.load()

# template = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are an AI which summarizes the text."),
#         ("human", "{data}"),
#     ]
# )

# prompt = template.format_messages(data=docs[0].page_content)



# model = ChatMistralAI(model="mistral-small-2506")

# result = model.invoke(prompt)

# print(result.content);

#-------------------------------------------------------------------------------

# code for pdf document loader
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

load_dotenv()


# file_path = Path(__file__).parent /"docuement loader/" "notes.txt"
# data = TextLoader(str(file_path))
data = PyPDFLoader("document loaders/notes.pdf")
docs = data.load()

template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI which summarizes the text."),
        ("human", "{data}"),
    ]
)

prompt = template.format_messages(data=docs[0].page_content)

model = ChatMistralAI(model="mistral-small-2506")

result = model.invoke(prompt)

print(result.content);



splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=100)

chunks = splitter.split_documents(docs);

print(len(chunks))