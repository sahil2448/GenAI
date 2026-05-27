# #  code for text document loader

# # from dotenv import load_dotenv
# # from langchain_mistralai import ChatMistralAI
# # from langchain_community.document_loaders import TextLoader
# # from langchain_core.prompts import ChatPromptTemplate
# # from pathlib import Path

# # load_dotenv()


# # # file_path = Path(__file__).parent /"docuement loader/" "notes.txt"
# # # data = TextLoader(str(file_path))
# # data = TextLoader("document loaders/notes.txt")
# # docs = data.load()

# # template = ChatPromptTemplate.from_messages(
# #     [
# #         ("system", "You are an AI which summarizes the text."),
# #         ("human", "{data}"),
# #     ]
# # )

# # prompt = template.format_messages(data=docs[0].page_content)



# # model = ChatMistralAI(model="mistral-small-2506")

# # result = model.invoke(prompt)

# # print(result.content);

# #-------------------------------------------------------------------------------

# # code for pdf document loader
# from dotenv import load_dotenv
# from langchain_mistralai import ChatMistralAI
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from pathlib import Path

# load_dotenv()


# # file_path = Path(__file__).parent /"docuement loader/" "notes.txt"
# # data = TextLoader(str(file_path))
# data = PyPDFLoader("document loaders/notes.pdf")
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



# splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=100)

# chunks = splitter.split_documents(docs);

# print(len(chunks))


# ---------------------Final complete RAG project starts from here -------------------

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = Chroma(
    persist_directory="chroma-db",
    embedding_function=embedding_model
)

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
    "k":4,
    "fetch_k":10, # we first fetch 10 documents based on similarity search and then we apply MMR to select the most relevant documents from those 10 documents. This is because MMR is a re-ranking algorithm that takes into account both relevance and diversity, so we need to have a larger set of documents to choose from in order to select the most relevant ones.
    "lambda_mult":0.5,# lambda_mult is a parameter that controls the trade-off between relevance and diversity in the MMR algorithm. A higher value of lambda_mult will give more weight to relevance, while a lower value will give more weight to diversity. In this case, we have set it to 0.5, which means that we are giving equal weight to both relevance and diversity when selecting the most relevant documents from the fetched set of documents.
    }
)

llm = ChatMistralAI(model = "mistral-small-2506")

#prompt template for question answering
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI which answers the question based on the given context..."
        "if answer is not found in the context, then say that the answer is not found in the context."),

        ("human", "Context: {context} \n\n Question: {question}"),
    ]
)

print("Rag system is ready to answer your questions...")

print("press 0 to exit the system")

while True:
    query = input("You: ")
    if query == "0":
        print("Exiting the system...")
        break
    docs = retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })

    response = llm.invoke(final_prompt)

    print("]\n AI: ", response.content)