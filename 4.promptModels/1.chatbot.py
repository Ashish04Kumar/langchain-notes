# from langchain_ollama import ChatOllama

# llm = ChatOllama(model="smollm2:360m")

# # Chat history
# history = []

# while True:
#     user_input = input("You: ")

#     if user_input.lower() == "exit":
#         break

#     # User message history mein store
#     history.append(("user", user_input))

#     # Puri history LLM ko bhejo
#     response = llm.invoke(history)

#     print("Bot:", response.content)

#     # AI response history mein store
#     history.append(("assistant", response.content))

from langchain_ollama import ChatOllama
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

llm = ChatOllama(model="qwen3-vl:2b")

history = [
    SystemMessage(
        content="You are a helpful chatbot. Remember the conversation history and answer questions based on it. If the user tells you their name, remember it."
    )
]

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    history.append(HumanMessage(content=user_input))

    response = llm.invoke(history)

    print("Bot:", response.content)

    history.append(AIMessage(content=response.content))