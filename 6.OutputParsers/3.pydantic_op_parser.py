from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, field_validator


class CustomerResponse(BaseModel):
    intent: str = Field(description="The intent of the user's message")
    status: str = Field(description="The status of the user's request")
    confidence: float = Field(description="Confidence score between 0 and 1")

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value):
        if not 0 <= value <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        return value

    @field_validator("status")
    @classmethod
    def validate_status(cls, value):
        allowed_status = [
            "pending",
            "processing",
            "completed",
            "failed"
        ]

        if value not in allowed_status:
            raise ValueError(
                f"Status must be one of {allowed_status}"
            )

        return value


llm = ChatOllama(model="smollm2:360m")

parser = PydanticOutputParser(
    pydantic_object=CustomerResponse
)

prompt = ChatPromptTemplate.from_template("""
You are a customer support assistant.

Return the answer according to this format:

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
print(result.intent)
print(result.status)
print(result.confidence)