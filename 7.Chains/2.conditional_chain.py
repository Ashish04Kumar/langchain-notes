from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

llm = ChatOllama(model="qwen3:4b")
# llm = ChatOllama(model="smollm2:360m")

parser = StrOutputParser()

# classification
class Feedback(BaseModel):
    feedback: str = Field(description='Give the feedback')
    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feeback text into positive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction': parser2.get_format_instructions()}
)



classifier_chain = prompt1 | llm | parser2

# result = print(classifier_chain.invoke({'feedback': 'This is a terrible smartphone'}))
# print(result.sentiment)

# branching

prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback :\n{feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback :\n{feedback}',
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', RunnableLambda(lambda x: {"feedback": x.feedback}) | prompt2 | llm | parser),
    (lambda x:x.sentiment == 'negative',RunnableLambda(lambda x: {"feedback": x.feedback}) | prompt3 | llm | parser),
    RunnableLambda(lambda x: "Could not find sentiment")
)

chain = classifier_chain | branch_chain 
print(chain.invoke({'feedback': 'This is a terrible phone'}))
chain.get_graph().print_ascii()