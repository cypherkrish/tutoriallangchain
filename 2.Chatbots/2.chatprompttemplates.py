from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

chat_tempalte = ChatPromptTemplate([
    ("system","You are a helpfull {domain} expert"),
    ("human","Explain in simple terms, the concept of {concept}")
]
)

prompt = chat_tempalte.invoke({
    "domain": "quantum physics",
    "concept": "quantum superposition"
}
)

print(prompt)

result = model.invoke(prompt)

print(result.content)