import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(page_title="GK Assistant", page_icon="✦", layout="centered")

# --------------------------------------------------------------------------
# Design tokens — "study lamp at night": ink navy + parchment + amber spark
# --------------------------------------------------------------------------
INK         = "#1B2A4A"
INK_SOFT    = "#2A3C61"
PARCHMENT   = "#F7F3E9"
PARCHMENT_2 = "#EFE8D8"
AMBER       = "#D9A441"
INK_MUTED   = "#5B6B8C"

LOGO_SVG = f"""
<svg width="40" height="40" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg">
  <circle cx="20" cy="20" r="20" fill="{INK}"/>
  <path d="M20 9 L23 17 L31 20 L23 23 L20 31 L17 23 L9 20 L17 17 Z" fill="{AMBER}"/>
</svg>
"""

st.markdown(
f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@500;600&family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
.stApp {{ background-color: {PARCHMENT}; }}
.gk-header {{
display: flex; align-items: center; gap: 14px;
padding: 18px 22px; background-color: {INK};
border-radius: 16px; margin-bottom: 28px;
box-shadow: 0 4px 14px rgba(27,42,74,0.18);
}}
.gk-header h1 {{
font-family: 'Fraunces', serif; font-weight: 600;
font-size: 22px; color: {PARCHMENT}; margin: 0;
}}
.gk-header p {{
font-size: 13px; color: {AMBER}; margin: 0;
letter-spacing: 0.4px; text-transform: uppercase;
}}
[data-testid="stChatMessage"] {{ padding: 4px 0; }}
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {{
flex-direction: row-reverse;
}}
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) .stChatMessageContent {{
background-color: {AMBER}; color: {INK};
border-radius: 16px 16px 4px 16px; padding: 10px 16px; font-weight: 500;
}}
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) .stChatMessageContent {{
background-color: {INK_SOFT}; color: {PARCHMENT};
border-radius: 16px 16px 16px 4px; padding: 10px 16px;
}}
[data-testid="stChatInput"] {{
background-color: {PARCHMENT_2}; border-radius: 14px; border: 1.5px solid {AMBER};
}}
[data-testid="stChatInput"] textarea {{ color: {INK}; }}
::-webkit-scrollbar {{ width: 8px; }}
::-webkit-scrollbar-thumb {{ background: {INK_MUTED}; border-radius: 8px; }}
</style>
<div class="gk-header">
{LOGO_SVG}
<div>
<h1>Chat AI · GK Assistant</h1>
<p>Ask anything · General Knowledge</p>
</div>
</div>""",
unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# LLM setup (uses Streamlit Cloud secret, not os.getenv)
# --------------------------------------------------------------------------
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=st.secrets["OPENROUTER_API_KEY"],
    model="liquid/lfm-2.5-1.2b-thinking:free",
    temperature=0,
)

SYSTEM_PROMPT = """You are a helpful, knowledgeable assistant.
- Answer clearly and directly; don't pad responses with unnecessary hedging.
- If you're not confident about a fact, say so explicitly rather than guessing.
- Keep responses concise unless the user asks for more detail.
- Ask a clarifying question only if the user's request is genuinely ambiguous."""

# --------------------------------------------------------------------------
# Chat state
# --------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm your GK Assistant. Ask me anything — history, science, current affairs, you name it."}
    ]

for msg in st.session_state.messages:
    avatar = "🧑" if msg["role"] == "user" else "✨"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# --------------------------------------------------------------------------
# Input + response
# --------------------------------------------------------------------------
if prompt := st.chat_input("Type your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="✨"):
        with st.spinner("Thinking..."):
            try:
                # Rebuild LangChain message history from session state
                lc_messages = [SystemMessage(content=SYSTEM_PROMPT)]
                for m in st.session_state.messages:
                    if m["role"] == "user":
                        lc_messages.append(HumanMessage(content=m["content"]))
                    else:
                        lc_messages.append(AIMessage(content=m["content"]))

                response = llm.invoke(lc_messages)
                answer = response.content
                st.markdown(answer)
            except Exception as e:
                answer = f"Sorry, something went wrong: {e}"
                st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
