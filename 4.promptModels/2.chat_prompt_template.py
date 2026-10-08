from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful teacher."),
    ("human", "Explain {topic} in simple language.")
])

llm = ChatOllama(model="qwen3:4b")

messages = prompt.format_messages(topic="RAG")

response = llm.invoke(messages)

print(response.content)