from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel, RunnableSequence
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate ,ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(model="smollm2:360m")
passthrough = RunnablePassthrough()
parser = StrOutputParser()

def word_counter(text):
    return len(text.split())


prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)



jon_gen_chain = RunnableSequence(prompt1, llm , parser)
parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count':  RunnableLambda(word_counter) #lambda x: len(x.split())
})

final_chain = RunnableSequence(jon_gen_chain , parallel_chain)
result = final_chain.invoke({
    'topic' :'AI'
})
print(result)