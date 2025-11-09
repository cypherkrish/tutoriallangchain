from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


#chat tempalte

chat_template = ChatPromptTemplate([
    ("system", "You are a helpfull customer support agent"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", '{query}')
])

chat_history = []
with open("chatbot_history.txt") as file:
    chat_history.extend(file.readline())

prompt = chat_template.invoke({
    "chat_history": chat_history,
    "query": "where is my order"
}
)

print(prompt)