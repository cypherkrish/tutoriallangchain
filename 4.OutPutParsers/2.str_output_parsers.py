from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

tempalte1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"])

template2 = PromptTemplate(
    template="Write a 4 point summary on the following {text}",
    input_variables=["text"])

parser = StrOutputParser()

# Chain
chain = tempalte1 | model | parser | template2 | model | parser

result = chain.invoke({"Indian Premier League 2023/2024"})

print(result)