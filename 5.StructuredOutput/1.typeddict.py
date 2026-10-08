from typing import TypedDict, Annotated, List, Optional, Literal

from langchain_ollama import ChatOllama


llm = ChatOllama(model="smollm2:360m")


class Review(TypedDict):

    summary: Annotated[
        str,
        "A brief summary of the review"
    ]

    sentiment: Annotated[
        Literal["positive", "negative", "neutral"],
        "Return sentiment as positive, negative, or neutral"
    ]

    rating: Annotated[
        Optional[int],
        "Optional rating from 1 to 5"
    ]

    pros: Annotated[
        List[str],
        "List of positive points mentioned in the review"
    ]


structured_model = llm.with_structured_output(Review)


result = structured_model.invoke(
    "The hardware is great, but the software feels bloated. "
    "There are too many pre-installed apps that I can't remove. "
    "Also, the UI looks outdated compared to other brands. "
    "Hoping for a software update to fix this."
)


print(result)