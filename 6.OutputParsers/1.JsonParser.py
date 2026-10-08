from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

llm = ChatOllama(model="qwen3:4b")

parser = JsonOutputParser()

prompt = ChatPromptTemplate.from_template("""
Return the answer in JSON format.

{format_instructions}

User message:
{query}
""")

chain = prompt | llm | parser

result = chain.invoke({
    "query": "Where is my refund?",
    "format_instructions": parser.get_format_instructions()
})

print(result)
print(type(result))



# schema enforce ni krskte json ouput parser me









# without chain

# prompt = ChatPromptTemplate.from_template("""
# Return the answer in JSON format.

# {format_instructions}

# User message:

# {query}
# """)

# prompt_value = prompt.invoke({
#     "query": "Where is my refund?",
#     "format_instructions": parser.get_format_instructions()
# })

# result = llm.invoke(prompt_value)

# print("LLM Response:")
# print(result)

# parsed_result = parser.invoke(result)

# print("Parsed Result:")
# print(parsed_result)

# print(type(parsed_result))