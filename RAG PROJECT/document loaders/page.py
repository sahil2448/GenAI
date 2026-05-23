from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import CharacterTextSplitter
url = "https://www.apple.com/mac/"

data = WebBaseLoader(url)

docs = data.load();



# print(len(docs));
print(docs[0].page_content)