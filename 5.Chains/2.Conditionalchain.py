from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_classic.schema.runnable import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

parser1 = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal["Positive", "Negative"] = Field(description='The sentiment of the feedback that the client provied')

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template="""
    Classify the sentiment of the following feedback text into Positive or Neative {feedback} and provide the response in the following format: {response_format}
    """,
    input_variables=['feedback'],
    partial_variables={'response_format': parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

prompt3 = PromptTemplate(
    template="Write an appropriate response to this Positive feedback {feedback}",
    input_variables=['feedback']
)

prompt4 = PromptTemplate(
    template="Write an appropriate response to this Negative feedback {feedback}",
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "Positive", prompt3 | model | parser1),
    (lambda x: x.sentiment == "Negative", prompt4 | model | parser1),
    RunnableLambda(lambda x: "No valid sentiment found")
)

chain = classifier_chain | branch_chain

result = chain.invoke({"feedback": "The phone is actually amazing"})

print(result)