import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

st.set_page_config(page_title="My First Agent", page_icon="🤖")
st.title("🤖 My First Agent")

SYSTEM_PROMPT = """You are a helpful, knowledgeable assistant.

- Answer clearly and directly; don't pad responses with unnecessary hedging.
- If you're not confident about a fact, say so explicitly rather than guessing.
- Keep responses concise unless the user asks for more detail.
- Ask a clarifying question only if the user's request is genuinely ambiguous."""

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model=os.getenv("MODEL_NAME"),
    temperature=float(os.getenv("MODEL_TEMPERATURE", 0)),
)

# Memory box — survives across re-runs, unlike normal variables
if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content=SYSTEM_PROMPT)]

# Redraw the whole conversation so far (since the script reruns each time)
for msg in st.session_state.messages[1:]:  # skip system prompt, don't show it
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# Wait for new user input
user_input = st.chat_input("Ask me anything...")

if user_input:
    st.session_state.messages.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = llm.invoke(st.session_state.messages)
                st.markdown(response.content)
                st.session_state.messages.append(AIMessage(content=response.content))
            except Exception as e:
                st.error("Something went wrong reaching the AI model. Please try again in a moment.")
                print(f"LLM call failed: {e}")  # shows up in your terminal, not the user's screen