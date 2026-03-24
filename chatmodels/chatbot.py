from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage

model = init_chat_model("mistral-large-2512",temperature=0.9,max_tokens=50);

    # To store the history, i am going to use a list to store the conversation history.
    # Because model is not able to remember the previous conversation, so we need to provide the conversation history as a context for the model to generate a response.
print("---------welcome type 0 to exit the application-----------")
messages = [
    SystemMessage(content="You are a funny AI agent")
]
while True:
    prompt = input("You : ")
    messages.append(HumanMessage(content = prompt))
    if prompt == "0":
        break

    response = model.invoke(messages)
    messages.append(AIMessage(content = response.content))
    print("Bot : " + response.content)


print(messages)