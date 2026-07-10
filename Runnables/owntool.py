from langchain.tools import tool

@tool
def get_greeting(name:str)->str:
    """Generate a greeting message for a user"""
    return f"Hello, {name}! Welcome to our platform. We hope you have a great experience!"

result = get_greeting.invoke({
    "name":"sahil"
})

print(result);
print(get_greeting.name)
print(get_greeting.description)
print(get_greeting.args)