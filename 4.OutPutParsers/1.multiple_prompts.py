from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

#list prompt
tempalte1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)

prompt1 = tempalte1.invoke({"topic":"Indian Premier League 2023/2024"})
result1 = model.invoke(prompt1)
print(result1)

print("**************************")
# Template 2

template2 = PromptTemplate(template="Write a 4 point summary on the following {text}",
                           input_variables=["text"])
prompt2 = template2.invoke({"text":str(result1)})

result2 = model.invoke(prompt2)

print(result2.content)


