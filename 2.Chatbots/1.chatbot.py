from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

chat_histrory = [
    SystemMessage(content="You are a helpfull assistant")
]

while True:
    user_input = input("User: ")
    chat_histrory.append(HumanMessage(content=user_input))
    if user_input.lower() == 'exit':
        break
    result = model.invoke(chat_histrory)
    chat_histrory.append(AIMessage(content=result.content))
    print("Bot: ", result.content)
    
print("Chat History: ", chat_histrory)