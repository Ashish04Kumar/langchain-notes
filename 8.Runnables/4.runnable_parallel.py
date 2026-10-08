from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate ,ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel , RunnableSequence
llm = ChatOllama(model="smollm2:360m")

prompt1 = PromptTemplate(
    template='Generate a tweet about {topic}',
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template='Generate a linked in post about {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet': RunnableSequence(prompt1, llm , parser),
    'linkedin': RunnableSequence(prompt2, llm , parser)
})

result = parallel_chain.invoke({
    'topic' : 'ai'
})

print(result)