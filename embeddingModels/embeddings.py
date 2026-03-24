from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

embeddings = OpenAIEmbeddings(
    model = 'text-embedding-3-large',
    dimensions = 64
)

vectors = embeddings.embed_query("You are going to learn GenAI.")

print(vectors)