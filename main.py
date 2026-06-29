import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()  # reads your .env file

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="liquid/lfm-2.5-1.2b-thinking:free",
    temperature=0,
)
user_question = input("Ask your question: ")
messages = [
    SystemMessage(content="""You are a helpful, knowledgeable assistant.

- Answer clearly and directly; don't pad responses with unnecessary hedging.
- If you're not confident about a fact, say so explicitly rather than guessing.
- Keep responses concise unless the user asks for more detail.
- Ask a clarifying question only if the user's request is genuinely ambiguous."""),
    
    HumanMessage(content=user_question),
]

response = llm.invoke(messages)
print(response.content)