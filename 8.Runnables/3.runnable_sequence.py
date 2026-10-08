from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate ,ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence 
llm = ChatOllama(model="qwen3:4b")

prompt = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='explain the following - {text}',
    input_variables=['text']
)
parser = StrOutputParser()
chain = RunnableSequence(prompt, llm, parser, prompt2, llm, parser)
print(chain.invoke({'topic': 'AI'}))