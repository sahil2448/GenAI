import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Load env variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="Funny Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .chat-container {
        max-width: 800px;
        margin: auto;
    }
    .user-msg {
        background-color: #3f3d45;
        color: white;
        padding: 10px 15px;
        border-radius: 12px;
        margin: 5px 0;
        text-align: left;
    }
    .bot-msg {
        padding: 10px 15px;
        border-radius: 12px;
        margin: 5px 0;
        text-align: left;
    }
</style>
""", unsafe_allow_html=True)

st.title("😂 Funny AI Chatbot")

# Initialize model (only once)
@st.cache_resource
def load_model():
    return init_chat_model("mistral-large-2512", temperature=0.9, max_tokens=100)

model = load_model()

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are a funny AI agent")
    ]

# Chat container
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Display chat history
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.markdown(f'<div class="user-msg">{msg.content}</div>', unsafe_allow_html=True)
    elif isinstance(msg, AIMessage):
        st.markdown(f'<div class="bot-msg">{msg.content}</div>', unsafe_allow_html=True)

# Input box (fixed at bottom)
prompt = st.chat_input("Type your message...")

if prompt:
    # Add user message
    st.session_state.messages.append(HumanMessage(content=prompt))

    # Display user instantly
    st.markdown(f'<div class="user-msg">{prompt}</div>', unsafe_allow_html=True)

    # Generate response
    with st.spinner("Thinking..."):
        response = model.invoke(st.session_state.messages)

    # Add AI response
    st.session_state.messages.append(AIMessage(content=response.content))

    # Display AI response
    st.markdown(f'<div class="bot-msg">{response.content}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    if st.button("Clear Chat"):
        st.session_state.messages = [
            SystemMessage(content="You are a funny AI agent")
        ]
        st.rerun()

    st.markdown("---")
    st.write("Built with ❤️ using Streamlit")
