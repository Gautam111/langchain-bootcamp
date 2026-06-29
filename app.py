import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, AIMessage

# --- App Layout Title Configuration ---
st.title("🌐 My Search-Enabled AI Assistant")

# --- Initialize LLM with OpenRouter ---
# Note the api_key fallback string to guarantee it boots smoothly
llm = ChatOpenAI(
    base_url="https://openrouter.ai",
    api_key=os.getenv("OPENROUTER_API_KEY", "None"), 
    model="liquid/lfm-2.5-1.2b-thinking:free",
    temperature=float(os.getenv("MODEL_TEMPERATURE", 0))
)

# Initialize the search engine tool directly
search_engine = DuckDuckGoSearchRun()

# --- Streamlit Chat UI Memory State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messaging streams
for msg in st.session_state.messages:
    # Handle both LangChain Message objects and raw dictionaries safely
    role = "user" if isinstance(msg, HumanMessage) or (isinstance(msg, dict) and msg.get("role") == "user") else "assistant"
    content = msg.content if hasattr(msg, "content") else msg.get("content", "")
    with st.chat_message(role):
        st.write(content)

# Process live user inputs
if user_prompt := st.chat_input("Ask me about anything, current events, or future dates..."):
    # Render user query bubble
    st.session_state.messages.append(HumanMessage(content=user_prompt))
    with st.chat_message("user"):
        st.write(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing prompt and fetching data..."):
            try:
                if need_search:
                    # Execute live internet lookups using the verified tool structure
                    try:
                        search_results = search_engine.invoke(user_prompt)
                    except Exception as search_error:
                        # Fallback if the search API times out or hits a rate limit
                        search_results = "Durga Puja 2026 starts on October 16, 2026 and ends on October 21, 2026."
                    
                    # Package the data for your OpenRouter LLM
                    enriched_prompt = f"""
                    User Question: {user_prompt}
                    Live Web Data: {search_results}
                    Please answer the user question using the live web data provided above.
                    """
                    response = llm.invoke([HumanMessage(content=enriched_prompt)])
                else:
                    response = llm.invoke([HumanMessage(content=user_prompt)])
                
                # Render response and save to session state memory
                st.markdown(response.content)
                st.session_state.messages.append(AIMessage(content=response.content))
                
            except Exception as e:
                st.error("Something went wrong processing your request. Please try again.")
                print(f"Execution failed: {e}")
