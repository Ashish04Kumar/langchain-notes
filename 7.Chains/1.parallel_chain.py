from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

llm = ChatOllama(model="smollm2:360m")

parser = StrOutputParser()

# Chain 1: Notes generate karega
prompt1 = PromptTemplate(
    template="Generate short notes from the following text:\n{text}",
    input_variables=["text"]
)

# Chain 2: Quiz generate karega
prompt2 = PromptTemplate(
    template="Generate 5 short questions from the following text:\n{text}",
    input_variables=["text"]
)

# Dono chains parallel mein chalengi
parallel_chain = RunnableParallel({
    "notes": prompt1 | llm | parser,
    "quiz": prompt2 | llm | parser
})

# result = parallel_chain.invoke({
#     "text": "Python is a high-level programming language used for web development, AI, automation and data science."
# })

# print(result)

prompt3 = PromptTemplate(
    template="""Merge these into one document:

Notes:
{notes}

Quiz:
{quiz}
""",
    input_variables=["notes", "quiz"]
)

final_chain = parallel_chain | prompt3 | llm | parser

result = final_chain.invoke({
    "text": "Python is a high-level programming language used for web development, AI, automation and data science."
})

print(result)
final_chain.get_graph().print_ascii()

"""
                         input
                           │
                           │
                  {"text": "Python..."}
                           │
                           ▼
                 ┌───────────────────┐
                 │  RunnableParallel │
                 └─────────┬─────────┘
                           │
             ┌─────────────┴─────────────┐
             │                           │
             ▼                           ▼
       ┌────────────┐              ┌────────────┐
       │  prompt1   │              │  prompt2   │
       │ Notes      │              │ Quiz       │
       └─────┬──────┘              └─────┬──────┘
             │                           │
             ▼                           ▼
       ┌────────────┐              ┌────────────┐
       │    llm     │              │    llm     │
       └─────┬──────┘              └─────┬──────┘
             │                           │
             ▼                           ▼
       ┌────────────┐              ┌────────────┐
       │   parser   │              │   parser   │
       └─────┬──────┘              └─────┬──────┘
             │                           │
             │                           │
             └─────────────┬─────────────┘
                           ▼
              {
                "notes": "...",
                "quiz": "..."
              }
                           │
                           ▼
                    ┌────────────┐
                    │  prompt3   │
                    │   Merge    │
                    └─────┬──────┘
                          │
                          ▼
                    ┌────────────┐
                    │    llm     │
                    └─────┬──────┘
                          │
                          ▼
                    ┌────────────┐
                    │   parser   │
                    └─────┬──────┘
                          │
                          ▼
                    FINAL OUTPUT
"""