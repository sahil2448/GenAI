import os
import shutil
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()

APP_TITLE = "RAG Assistant"
BASE_PERSIST_DIR = "chroma-db"
UPLOAD_DIR = Path("uploaded_docs")
UPLOAD_DIR.mkdir(exist_ok=True)


# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
        .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
        .hero {
            padding: 1.4rem 1.5rem;
            border-radius: 1.25rem;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 45%, #334155 100%);
            color: white;
            margin-bottom: 1rem;
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.25);
        }
        .hero h1 {margin: 0; font-size: 2.1rem;}
        .hero p {margin: 0.45rem 0 0 0; opacity: 0.9; font-size: 1rem;}
        .card {
            border: 1px solid rgba(148,163,184,0.25);
            border-radius: 1rem;
            padding: 1rem 1rem;
            background: rgba(248,250,252,0.7);
        }
        .small-note {font-size: 0.9rem; color: #475569;}
        .stChatMessage {border-radius: 1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>📚 RAG Assistant</h1>
        <p>Upload a PDF, build a searchable knowledge base, and ask questions in a clean chat interface.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Session state
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Upload a PDF from the sidebar, build the knowledge base, and ask me questions about it.",
        }
    ]

if "vector_store_ready" not in st.session_state:
    st.session_state.vector_store_ready = False

if "current_doc_name" not in st.session_state:
    st.session_state.current_doc_name = ""

if "persist_dir" not in st.session_state:
    st.session_state.persist_dir = BASE_PERSIST_DIR


# -----------------------------
# Helpers
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatMistralAI(model="mistral-small-2506")


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful AI assistant. Answer only from the given context. "
            "If the answer is not in the context, say clearly that the answer was not found in the uploaded document.",
        ),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)


def save_uploaded_file(uploaded_file) -> Path:
    """Save uploaded file to local disk and return its path."""
    target_path = UPLOAD_DIR / uploaded_file.name
    with open(target_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return target_path


def build_vector_store_from_pdf(pdf_path: str, persist_dir: str = BASE_PERSIST_DIR):
    """Load PDF, split into chunks, embed, and persist in Chroma."""
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embedding_model = get_embedding_model()
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_dir,
    )
    return vector_store, len(chunks)


def load_vector_store(persist_dir: str = BASE_PERSIST_DIR):
    embedding_model = get_embedding_model()
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_model,
    )


def get_retriever(vector_store):
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5,
        },
    )


@st.cache_data(show_spinner=False)
def get_file_size_mb(path: str) -> float:
    return round(os.path.getsize(path) / (1024 * 1024), 2)


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Control Panel")
    st.caption("Build your document memory and chat with it.")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    col_a, col_b = st.columns(2)
    with col_a:
        build_btn = st.button("Build / Refresh", use_container_width=True)
    with col_b:
        clear_btn = st.button("Clear Chat", use_container_width=True)

    st.divider()
    st.subheader("Status")
    st.write(f"**Knowledge base:** {'Ready' if st.session_state.vector_store_ready else 'Not built yet'}")
    st.write(f"**Active file:** {st.session_state.current_doc_name or 'None'}")
    st.write(f"**Persist dir:** `{st.session_state.persist_dir}`")

    st.divider()
    st.subheader("How it works")
    st.write(
        "1. Upload a PDF\n"
        "2. Build the vector store\n"
        "3. Ask questions in chat\n"
        "4. See retrieved chunks for transparency"
    )


# -----------------------------
# Handle clear chat
# -----------------------------
if clear_btn:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Chat cleared. Upload a PDF and build the knowledge base to begin again.",
        }
    ]
    st.rerun()


# -----------------------------
# Build / refresh vector store
# -----------------------------
if build_btn:
    try:
        if uploaded_file is None:
            st.warning("Please upload a PDF first.")
        else:
            with st.spinner("Saving PDF and building embeddings..."):
                saved_path = save_uploaded_file(uploaded_file)

                # Use a file-specific persistence directory so each uploaded doc can be isolated.
                safe_name = Path(uploaded_file.name).stem.replace(" ", "_")
                doc_persist_dir = f"chroma-db-{safe_name}"
                if os.path.exists(doc_persist_dir):
                    shutil.rmtree(doc_persist_dir)

                vector_store, chunk_count = build_vector_store_from_pdf(
                    pdf_path=str(saved_path),
                    persist_dir=doc_persist_dir,
                )

                st.session_state.persist_dir = doc_persist_dir
                st.session_state.vector_store_ready = True
                st.session_state.current_doc_name = uploaded_file.name
                st.session_state.vector_store = vector_store
                st.session_state.retriever = get_retriever(vector_store)

                st.success(f"Knowledge base ready. {chunk_count} chunks indexed from {uploaded_file.name}.")
    except Exception as e:
        st.error(f"Failed to build the knowledge base: {e}")


# -----------------------------
# Default fallback: load existing DB if available
# -----------------------------
if "retriever" not in st.session_state and os.path.exists(BASE_PERSIST_DIR):
    try:
        vector_store = load_vector_store(BASE_PERSIST_DIR)
        st.session_state.vector_store = vector_store
        st.session_state.retriever = get_retriever(vector_store)
        st.session_state.vector_store_ready = True
        st.session_state.current_doc_name = st.session_state.current_doc_name or "Existing Chroma DB"
    except Exception:
        pass


# -----------------------------
# Main content layout
# -----------------------------
left, right = st.columns([1.4, 1])

with left:
    st.subheader("Chat")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_query = st.chat_input("Ask a question about your document...")

with right:
    st.subheader("Document Preview")
    st.markdown(
        """
        <div class="card">
            <strong>What you get</strong><br><br>
            • PDF upload support<br>
            • Persistent Chroma vector store<br>
            • MMR retrieval for better context coverage<br>
            • Clean chat UX with answer transparency
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown(
        "<p class='small-note'>Tip: Ask specific questions for better answers. "
        "For example: <i>What does the document say about billing terms?</i></p>",
        unsafe_allow_html=True,
    )

    if st.session_state.vector_store_ready:
        st.success("Vector store is ready.")
    else:
        st.info("Upload a PDF and click Build / Refresh.")


# -----------------------------
# Answer generation
# -----------------------------
if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})

    if "retriever" not in st.session_state:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "No knowledge base is loaded yet. Upload a PDF and build the index first.",
            }
        )
        st.rerun()

    try:
        with st.spinner("Thinking..."):
            docs = st.session_state.retriever.invoke(user_query)
            context = "\n\n".join([doc.page_content for doc in docs])

            final_prompt = prompt.invoke(
                {
                    "context": context,
                    "question": user_query,
                }
            )

            llm = get_llm()
            response = llm.invoke(final_prompt)
            answer = response.content

        st.session_state.messages.append({"role": "assistant", "content": answer})

        with st.expander("Retrieved context"):
            for i, doc in enumerate(docs, start=1):
                st.markdown(f"**Chunk {i}**")
                st.write(doc.page_content[:1200])
                st.caption(f"Source metadata: {doc.metadata}")
                st.divider()

        st.rerun()

    except Exception as e:
        st.session_state.messages.append(
            {"role": "assistant", "content": f"Something went wrong while answering: {e}"}
        )
        st.rerun()


# -----------------------------
# Footer
# -----------------------------
st.divider()
st.caption("Built with Streamlit + LangChain + Chroma + Mistral")
