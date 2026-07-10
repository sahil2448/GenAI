# load all the libraries 
from dotenv import load_dotenv
load_dotenv()

import os
import requests
from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage,ToolMessage
from tavily import TavilyClient 
from rich import print
# now let's create a tool
@tool
def get_weather(city:str)->str:
    """Get current weather of a city"""
    API_KEY = os.getenv("OPENWEATHER_API_KEY")
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()

    print("DEBUG:",data)

    if str(data.get("cod")) != "200":
        return f"Error: {data.get('message', 'Unable to fetch weather data')}"

    temp = data['main']['temp']
    desc = data['weather'][0]['description']

    return f"Weather in {city}: {temp}°C, {desc}"

print(get_weather.invoke("Kolhapur"))

# Another tool - Tavily news tool
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def get_news(city:str)->str:
    """Get latest news about the city"""

    response = tavily_client.search(
        query=f"latest news in {city}",
        search_depth="basic",
        max_results=3
    )

    results = response.get("results",[])

    if not results:
        return f"No news found for {city}"
    
    news_list = []

    for r in results:
        title = r.get("title","no title")
        url = r.get("url","")
        snippet = r.get("content","")

        news_list.append(
            f"-{title}\n .. {url}\n .. {snippet[:100]}..."
        )

    return f"Latest news in {city}:\n\n" + "\n\n".join(news_list)

print(get_news.invoke("kolhapur"))

llm = ChatMistralAI(model="mistral-small-2506")

tools = {
    "get_weather":get_weather,
    "get_news":get_news
}

llm_with_tool = llm.bind_tools([get_weather,get_news])

# agent loop - very important

messages = []

print("City Intelligence System")
print("Type Exit to quit")

while True:
    user_input = input("You : ")
    if user_input.lower() == "exit":
        break
    messages.append(HumanMessage(content=user_input))

    while True:
        result = llm_with_tool.invoke(messages)
        messages.append(result)

        #if tool is required

        if result.tool_calls:
            for tool_call in result.tool_calls:
                tool_name = tool_call['name']

                #HUMAN IN THE LOOP
                confirm = input(f"Do you want to call the tool '{tool_name}'? (yes/no): ")

                if confirm.lower() == "no":
                    print("Tool call denied.")
                    break

                # execute the tool
                tool_result = tools[tool_name].invoke(tool_call)
                messages.append(ToolMessage(
                                name=tool_name,
                                content=tool_result,
                                tool_call_id=tool_call['id']
                            )
                    )
        
        else:
            print(result.content)
            break