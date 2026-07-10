################---------------Tool Calling Example Part1 -----------------------------------------
# from dotenv import load_dotenv
# load_dotenv()

# from langchain_mistralai import ChatMistralAI
# from langchain.tools import tool
# from rich import print

# #1 creating a tool
# @tool
# def get_text_length(text:str)->int:
#     """Returns the number of Characters in a given string"""
#     return len(text)

# llm = ChatMistralAI(model="mistral-small-2506")

# # Tool binding
# llm_with_tool = llm.bind_tools([get_text_length])

# # result = llm.invoke("Hello")
# # result2 = llm_with_tool.invoke("Hello")
# # result = llm.invoke("Returns the number of Characters in a given string: 'How are you'")
# # result2 = llm_with_tool.invoke("Returns the number of Characters in a given string: 'How are you'")

# # # print(result.content)
# # print(result);
# # print("----------------------------------------------------------")
# # print(result2);


# result = llm_with_tool.invoke("Returns the number of Characters in a given string: 'How are you'")
# # print(result)
# # print(result.tool_calls[0])

# if result.tool_calls:
#     tool_call = result.tool_calls[0]

# tool_name = tool_call['name'];
# tool_args = tool_call['args'];
# # print(tool_name)
# # print(tool_args)

# tool_result = get_text_length.invoke(tool_args)

# # print(tool_result)

# final_response = llm.invoke(f"The length of the string is: {tool_result}")

# # print(final_response.content)

# # print(result.tool_calls[0])
# print(get_text_length.invoke({
#     'name': 'get_text_length',
#     'args': {'text': 'How are you'},
#     'id': 'ZYzi5DoKz',
#     'type': 'tool_call'
# })) #ToolMessage(content='11', name='get_text_length', tool_call_id='ZYzi5DoKz') -> result for this print statement


################--------------- Part2 -----------------------------------------
#-------We want to maintain a proper history to handle multi-turn conversations

from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from rich import print
from langchain_core.messages import HumanMessage


#1 creating a tool
@tool
def get_text_length(text:str)->int:
    """Returns the number of Characters in a given string"""
    return len(text)

llm = ChatMistralAI(model="mistral-small-2506")

# Tool binding
llm_with_tool = llm.bind_tools([get_text_length]) 

tools = {
    'get_text_length': get_text_length
}
message = []
query = HumanMessage("Returns the number of Characters in a given string : 'How are you'")
message.append(query)
# print(message)


result = llm_with_tool.invoke(message)
message.append(result)

# print(message)

if result.tool_calls:
    tool_name = result.tool_calls[0]['name']
    tool_message = tools[tool_name].invoke(result.tool_calls[0])
    message.append(tool_message)
    # print(message)


result = llm_with_tool.invoke(message)
print(result.content)