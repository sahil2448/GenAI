from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser

# tools are also runnables, so we can use them in a chain
search_tool = TavilySearchResults(max_results = 5) # max result to return
llm = ChatMistralAI(model = "mistral-small-2506")

prompt = ChatPromptTemplate.from_template(
"""
You are a very helpful assistant

Summarise the following news search results in simple bullet points, and provide a summary of the overall topic in 1-2 lines.

{news}
"""
)

chain = prompt | llm | StrOutputParser()

news_result = search_tool.run("latest AI news of 2026");

result = chain.invoke({
    "news":news_result
})

print(result);