from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
)

model = ChatHuggingFace(llm=llm)

response = model.invoke("What are your thoughts on US,Israil and iran war currently going on?")

print(response.content)