from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model

# model = init_chat_model("groq:meta-llama/llama-4-scout-17b-16e-instruct")

# model = init_chat_model("gpt-5") // Not available yet, but will be in the future.
model = init_chat_model("mistral-large-2512",temperature=0.7,max_tokens=20);
#  temperature is a parameter that controls the randomness of the model's output. A higher temperature will result in more random and creative responses, while a lower temperature will produce more deterministic and focused responses. Setting it to 0 means the model will always choose the most likely next word, resulting in very deterministic output.
#  if want more creativity - > temperature=0.7 or 1.0
# if want more focused and deterministic output - > temperature=0.2 or 0.3
# print(model)

response = model.invoke("give me a paragraph about the importance of AI in healthcare")
print(response.content)