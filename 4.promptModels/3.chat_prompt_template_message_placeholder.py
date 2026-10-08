from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage


# 1. LLM
llm = ChatOllama(model="qwen3:4b")


# 2. Chat Prompt Template
chat_template = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful chatbot. Remember the conversation."
    ),

    # Yahan history runtime par insert hogi
    MessagesPlaceholder(variable_name="history"),

    # Current user question
    ("human", "{question}")
])


# 3. Chat history
history = []


# 4. Chat loop
while True:

    question = input("You: ")

    if question.lower() == "exit":
        break

    # 5. Template ko data do
    prompt_value = chat_template.invoke({
        "history": history,
        "question": question
    })

    # 6. LLM ko messages bhejo
    response = llm.invoke(prompt_value)

    print("Bot:", response.content)

    # 7. History mein user + AI message save karo
    history.append(
        HumanMessage(content=question)
    )

    history.append(
        AIMessage(content=response.content)
    )