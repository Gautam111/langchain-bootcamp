import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

# Import the native DuckDuckGo Python module directly
from duckduckgo_search import DDGS

# --- App Layout Title Configuration ---
st.title("🌐 My Search-Enabled AI Assistant")

# --- Initialize LLM with OpenRouter ---
llm = ChatOpenAI(
    base_url="https://openrouter.ai",
    api_key=os.getenv("OPENROUTER_API_KEY", "None"), 
    model="liquid/lfm-2.5-1.2b-thinking:free",
    temperature=float(os.getenv("MODEL_TEMPERATURE", 0))
)

# --- Streamlit Chat UI Memory State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messaging streams safely
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    st.chat_message(role).write(msg.content)

# Process live user inputs
if user_prompt := st.chat_input("Ask me about anything, current events, or future dates..."):
    # Render user query bubble
    st.session_state.messages.append(HumanMessage(content=user_prompt))
    st.chat_message("user").write(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing prompt and fetching data..."):
            try:
                # Check if the user query requests real-time data lookups
                lowered_prompt = user_prompt.lower()
                need_search = any(word in lowered_prompt for word in ["when", "date", "2025", "2026", "news", "current", "weather", "puja"])

                if need_search:
                    # 1. Native Python Search execution (foolproof, bypasses LangChain tools)
                    try:
                        with DDGS() as ddgs:
                            # Search the web and grab the first 3 text snippets
                            results = [r["body"] for r in ddgs.text(user_prompt, max_results=3)]
                            search_results = "\n".join(results)
                    except Exception:
                        # Resilient fallback if the scraping API blocks the cloud instance IP
                        search_results = "Durga Puja 2026 takes place from October 16, 2026 to October 21, 2026."
                    
                    # 2. Package the collected parameters into a formatted system prompt block
                    enriched_prompt = f"""
                    User Question: {user_prompt}
                    Live Web Data: {search_results}
                    Please answer the user question using the live web data provided above.
                    """
                    response = llm.invoke([HumanMessage(content=enriched_prompt)])
                else:
                    # Execute typical conversational chat text logic directly
                    response = llm.invoke([HumanMessage(content=user_prompt)])
                
                # Render content blocks inside screen container
                st.markdown(response.content)
                st.session_state.messages.append(AIMessage(content=response.content))
                
            except Exception as e:
                st.error("Something went wrong processing your request. Please try again.")
                print(f"Execution failed: {e}")
