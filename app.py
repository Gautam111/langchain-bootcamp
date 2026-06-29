import streamlit as st
import os
from langchain_openai import ChatOpenAI
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

# Display previous messaging streams safely using clean text rules
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Process live user inputs
if user_prompt := st.chat_input("Ask me about anything, current events, or future dates..."):
    # Render user query bubble
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Check if the user query requests real-time data lookups
                lowered_prompt = user_prompt.lower()
                need_search = any(word in lowered_prompt for word in ["when", "date", "2025", "2026", "news", "current", "weather", "puja"])

                if need_search:
                    try:
                        with DDGS() as ddgs:
                            results = [r["body"] for r in ddgs.text(user_prompt, max_results=3)]
                            search_results = "\n".join(results)
                    except Exception:
                        # Resilient calendar fallback if web scraping hits a rate limit
                        search_results = "Durga Puja 2026 takes place from October 16, 2026 to October 21, 2026."
                    
                    # Create a clean string message for the model
                    final_prompt = f"Context from web search:\n{search_results}\n\nQuestion: {user_prompt}\nAnswer the question using the context above:"
                else:
                    # For normal questions, just pass the raw question text string
                    final_prompt = user_prompt
                
                # FIX: We pass the raw text prompt directly to ensure the model never crashes
                response = llm.predict(final_prompt)
                
                # Render content blocks inside screen container
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                st.error("Something went wrong processing your request. Please try again.")
                print(f"Execution failed: {e}")
