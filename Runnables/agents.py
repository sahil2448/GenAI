from dotenv import load_dotenv
load_dotenv()

import os
import requests

from rich import print

from tavily import TavilyClient

from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_mistralai import ChatMistralAI


# ==========================================================
# Environment Variables
# ==========================================================

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not OPENWEATHER_API_KEY:
    raise ValueError("OPENWEATHER_API_KEY not found.")

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY not found.")


tavily_client = TavilyClient(api_key=TAVILY_API_KEY)


# ==========================================================
# Tools
# ==========================================================

@tool
def get_weather(city: str) -> str:
    """
    Get the current weather of a city.
    """

    url = (
        "https://api.openweathermap.org/data/2.5/weather"
        f"?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    )

    try:
        response = requests.get(url, timeout=10)
        data = response.json()

        if str(data.get("cod")) != "200":
            return f"Weather Error: {data.get('message')}"

        temperature = data["main"]["temp"]
        description = data["weather"][0]["description"]

        return (
            f"Current weather in {city}:\n"
            f"Temperature : {temperature}°C\n"
            f"Condition   : {description}"
        )

    except Exception as e:
        return f"Weather API Error: {e}"


@tool
def get_news(city: str) -> str:
    """
    Get latest news about a city.
    """

    try:
        response = tavily_client.search(
            query=f"Latest news in {city}",
            search_depth="basic",
            max_results=3,
        )

        results = response.get("results", [])

        if not results:
            return f"No news found for {city}."

        news = []

        for i, article in enumerate(results, start=1):

            title = article.get("title", "No Title")
            url = article.get("url", "")
            content = article.get("content", "")

            news.append(
                f"""{i}.
Title : {title}
URL   : {url}
Info  : {content[:150]}...
"""
            )

        return "\n".join(news)

    except Exception as e:
        return f"News API Error: {e}"


# ==========================================================
# LLM
# ==========================================================

llm = ChatMistralAI(model="mistral-small-2506")

TOOLS = {
    "get_weather": get_weather,
    "get_news": get_news,
}

llm_with_tools = llm.bind_tools(list(TOOLS.values()))


# ==========================================================
# Agent Loop
# ==========================================================

def run_agent():

    messages = []

    print("=" * 60)
    print("[bold cyan]City Intelligence System[/bold cyan]")
    print("Ask about weather or city news.")
    print("Type [bold red]exit[/bold red] to quit.")
    print("=" * 60)

    while True:

        user_input = input("\nYou : ")

        if user_input.lower() == "exit":
            print("\nGoodbye!")
            break

        messages.append(HumanMessage(content=user_input))

        while True:

            ai_response = llm_with_tools.invoke(messages)
            messages.append(ai_response)

            if not ai_response.tool_calls:
                print(f"\nAssistant : {ai_response.content}")
                break

            for tool_call in ai_response.tool_calls:

                tool_name = tool_call["name"]

                permission = input(
                    f"\nAllow tool '{tool_name}'? (yes/no): "
                ).strip().lower()

                if permission != "yes":
                    print("Tool execution cancelled.")
                    continue

                tool_result = TOOLS[tool_name].invoke(tool_call)

                print(f"\n[green]Tool Output[/green]\n{tool_result}")

                messages.append(
                    ToolMessage(
                        tool_call_id=tool_call["id"],
                        name=tool_name,
                        content=tool_result,
                    )
                )


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":
    run_agent()