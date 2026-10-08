from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableSequence,
    RunnableBranch,
)
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(model="smollm2:360m")
passthrough = RunnablePassthrough()
parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Write a detailed report on {topic}", input_variables=["topic"]
)


prompt2 = PromptTemplate(
    template="Summarise the following text \n {text}", input_variables=["text"]
)

report_generation_chain = RunnableSequence(prompt1, llm, parser)
branch_chain = RunnableBranch(
    (lambda x: len(x.split()) < 900, RunnableSequence(prompt2, llm, parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_generation_chain, branch_chain)
result = final_chain.invoke({
    'topic': 'Russia vs Ukraine'
})
print(result)