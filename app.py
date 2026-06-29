import os
import streamlit as streamlit  # Standard convention is usually 'import streamlit as st'
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Load environment variables
load_dotenv()

# App layout and title
streamlit.set_page_config(page_title="AI Assistant", page_icon="🤖")
streamlit.title("🤖 Chat AI Assistant")

# Initialize the Language Model
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="liquid/lfm-2.5-1.2b-thinking:free",
    temperature=0,
)

# Text area for user input
user_question = streamlit.text_area("Ask your question:", placeholder="Type something here...")

# Trigger API call when user clicks the button
if streamlit.button("Submit"):
    if user_question.strip() == "":
        streamlit.warning("Please enter a question before submitting.")
    else:
        # Show a loading spinner during the API request
        with streamlit.spinner("Thinking..."):
            messages = [
                SystemMessage(content="""You are a helpful, knowledgeable assistant.
- Answer clearly and directly; don't pad responses with unnecessary hedging.
- If you're not confident about a fact, say so explicitly rather than guessing.
- Keep responses concise unless the user asks for more detail.
- Ask a clarifying question only if the user's request is genuinely ambiguous."""),
                HumanMessage(content=user_question),
            ]
            
            try:
                response = llm.invoke(messages)
                
                # Display the response in a neat markdown block
                streamlit.subheader("Response:")
                streamlit.markdown(response.content)
                
            except Exception as e:
                streamlit.error(f"An error occurred: {e}")
