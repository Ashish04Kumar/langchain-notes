from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema

llm = ChatOllama(model="smollm2:360m")

response_schemas = [
    ResponseSchema(
        name="intent",
        description="The intent of the user's message"
    ),
    ResponseSchema(
        name="status",
        description="The status of the user's request"
    )
]

parser = StructuredOutputParser.from_response_schemas(response_schemas)

prompt = ChatPromptTemplate.from_template("""
You are a customer support assistant.

Return the answer according to the following format:

{format_instructions}

User message:
{query}
""")

prompt_value = prompt.invoke({
    "query": "Where is my refund?",
    "format_instructions": parser.get_format_instructions()
})

result = llm.invoke(prompt_value)

parsed_result = parser.invoke(result)

print(parsed_result)
print(type(parsed_result))