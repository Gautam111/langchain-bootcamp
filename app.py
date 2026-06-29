import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, AIMessage

# Modern, correct agent imports
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
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
    model="liquid/lfm-2.5-1.2b-thinking:free",
    temperature=float(os.getenv("MODEL_TEMPERATURE", 0)),
)
search_tool = DuckDuckGoSearchRun()
tools = [search_tool]
search_tool = DuckDuckGoSearchRun()
tools = [search_tool]

# 2. Pull a standard reasoning prompt template from the LangChain hub
prompt = hub.pull("hwchase17/react")

# 3. Create the modern ReAct agent structure
agent = create_react_agent(llm, tools, prompt)

# 4. Create the final executor runtime environment
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True,
    handle_parsing_errors=True
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
        with st.spinner("Searching the web and thinking..."):
            try:
                # 1. Changed llm.invoke to agent_executor.run (and pass text string, not the full chat history list)
                response_text = agent_executor.run(user_prompt) 
                
                # 2. Markdown now renders the raw text string returned by the agent
                st.markdown(response_text)
                
                # 3. Append the response text cleanly to your AIMessage object
                st.session_state.messages.append(AIMessage(content=response_text))
            except Exception as e:
                st.error("Something went wrong reaching the AI model. Please try again in a moment.")
                print(f"Agent loop execution failed: {e}")
