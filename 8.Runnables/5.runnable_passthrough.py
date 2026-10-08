from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate ,ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel , RunnableSequence, RunnablePassthrough

llm = ChatOllama(model="smollm2:360m")
passthrough = RunnablePassthrough()
parser = StrOutputParser()

prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='explain the following - {text}',
    input_variables=['text']
)

joke_gen_chain = RunnableSequence(prompt1, llm, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(prompt2 | llm | parser)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)
result=final_chain.invoke({'topic' : 'cricket'})
print(result)