from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class Review(TypedDict):
    summary: str
    sentiment: str
    
structured_model = model.with_structured_output(Review)

prompt = """
The hardware is greate, but the software feels kind of boalted. So many boilderplate apps and my phone keeps hanging when I play PUBG 
"""

response = structured_model.invoke(prompt)

print(response)

#============

new_prompt = f"Generate sentiment and summary of the review given. The review is {prompt}"
result = model.invoke(new_prompt)
print(result.content)