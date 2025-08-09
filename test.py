import os
import tempfile
from typing import List, Tuple
from dotenv import load_dotenv

import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

from langchain_groq import ChatGroq

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage


# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="BuddyBot - Your Office Buddy", page_icon="🤝", layout="wide")

# -----------------------------
# Sidebar: Settings
# -----------------------------
with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("Configure your BuddyBot")

    # API Key
    default_api_key = os.getenv("GROQ_API_KEY", "")
    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=default_api_key,
        help="Paste your Groq API key. It will be used in-memory for this session.",
    )
    if groq_api_key:
        os.environ["GROQ_API_KEY"] = "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"

    model = st.selectbox(
        "Model",
        options=[
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
        ],
        index=0,
    )
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.2, value=0.5, step=0.1)
    max_tokens = st.slider("Max tokens", min_value=64, max_value=2048, value=512, step=64)

    st.markdown("---")
    st.subheader("Retrieval")
    top_k = st.slider("Top K chunks", min_value=1, max_value=10, value=4, step=1)
    score_threshold = st.slider(
        "Auto-route threshold (similarity)",
        min_value=0.0, max_value=1.0, value=0.35, step=0.05,
        help="Higher means stricter. If best match is below this, BuddyBot will switch to general chat."
    )
    mode = st.radio(
        "Answering mode",
        options=["Auto", "Docs (RAG)", "General chat"],
        index=0,
        help="Auto decides between RAG and general chat based on similarity.",
    )
    show_citations = st.checkbox("Show citations", value=True)

    st.markdown("---")
    uploaded_files = st.file_uploader(
        "Upload PDFs to extend knowledge",
        type=["pdf"],
        accept_multiple_files=True,
        help="Add more PDFs at any time; they’ll be indexed for this session.",
    )

    if st.button("Clear chat"):
        st.session_state.pop("messages", None)
        st.session_state.pop("general_history", None)
        st.session_state.pop("rag_memory", None)
        st.session_state.pop("vectorstore", None)
        st.success("Cleared chat and memory.")


# -----------------------------
# Header
# -----------------------------
col1, col2 = st.columns([1, 5])
with col1:
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.image("bot.png", width=60)
with col2:
    st.markdown(
        "<span style='font-size: 2.2em; font-weight: bold; margin-bottom: 12px; display: inline-block;'>BuddyBot — Your Office Buddy</span><br>"
        "<span style='color:#666;'>Chat naturally. When needed, I’ll reference your PDFs with citations.</span>",
        unsafe_allow_html=True
    )


# -----------------------------
# Initialize globals (once)
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_embeddings():
    # Lightweight, fast embeddings
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")

@st.cache_resource(show_spinner=False)
def init_vectorstore():
    # Ephemeral in-memory Chroma (no persistent dir by default)
    return None  # We'll create it once we have docs

def ensure_vectorstore(vs, docs):
    # Create or extend a Chroma store with provided docs
    embeddings = get_embeddings()
    if vs is None and docs:
        return Chroma.from_documents(docs, embedding=embeddings, collection_name="buddybot")
    if vs is not None and docs:
        vs.add_documents(docs)
        return vs
    return vs

def load_pdfs_to_docs(files) -> List:
    docs = []
    for f in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.read())
            tmp.flush()
            loader = PyPDFLoader(tmp.name)
            docs.extend(loader.load())
    return docs

def chunk_docs(docs, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)


# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "general_history" not in st.session_state:
    st.session_state.general_history = []  # list[HumanMessage|AIMessage]

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = init_vectorstore()

if "rag_memory" not in st.session_state:
    # Memory specifically for the RAG chain
    st.session_state.rag_memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

# Seed initial file (optional)
default_pdf = "AnswersNewComer.pdf"
if st.session_state.vectorstore is None and os.path.exists(default_pdf):
    base_docs = PyPDFLoader(default_pdf).load()
    chunks = chunk_docs(base_docs)
    st.session_state.vectorstore = ensure_vectorstore(st.session_state.vectorstore, chunks)

# Handle uploads
if uploaded_files:
    new_docs = load_pdfs_to_docs(uploaded_files)
    new_chunks = chunk_docs(new_docs)
    st.session_state.vectorstore = ensure_vectorstore(st.session_state.vectorstore, new_chunks)
    st.success(f"Indexed {len(new_chunks)} chunks from {len(uploaded_files)} file(s).")


# -----------------------------
# LLMs & Chains
# -----------------------------
def get_llm():
    return ChatGroq(model=model, temperature=temperature, max_tokens=max_tokens)

# General chat chain (no retrieval)
SYSTEM_PROMPT = """You are BuddyBot, a warm, concise, and helpful office buddy.
- Be direct and clear.
- If user asks about internal policies, first check if 'Docs (RAG)' was used. In general chat, answer broadly and invite switching to Docs for exact citations.
- Ask clarifying questions when needed.
- Use markdown for readability.
"""

general_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

def general_chat(input_text: str) -> str:
    chain = general_prompt | get_llm()
    resp = chain.invoke({"chat_history": st.session_state.general_history, "input": input_text})
    return resp.content

# RAG chain
def get_rag_chain():
    if st.session_state.vectorstore is None:
        return None
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": top_k})
    llm = get_llm()
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=st.session_state.rag_memory,
        return_source_documents=True
    )

def auto_should_use_rag(query: str) -> Tuple[bool, float]:
    """
    Simple heuristic using similarity score from vectorstore.
    Notes:
      - Depending on vectorstore config, scores may be similarity (higher is better) or distance (lower is better).
      - With Chroma's default, scores from similarity_search_with_score are usually higher-is-better.
      - Tune 'score_threshold' in sidebar if routing feels off.
    """
    if st.session_state.vectorstore is None:
        return False, 0.0
    try:
        results = st.session_state.vectorstore.similarity_search_with_score(query, k=1)
        if not results:
            return False, 0.0
        _, score = results[0]
        return (score >= score_threshold), float(score)
    except Exception:
        # Fall back safely to general chat
        return False, 0.0


# -----------------------------
# Chat UI
# -----------------------------
# Render history
for m in st.session_state.messages:
    st.chat_message(m["role"]).markdown(m["content"])

prompt = st.chat_input("Ask anything… I’ll route to docs when helpful.")

if prompt:
    # Show user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Decide mode
    use_rag = False
    route_note = ""
    if mode == "Docs (RAG)":
        use_rag = True
        route_note = ""
    elif mode == "General chat":
        use_rag = False
        route_note = ""
    else:
        use_rag, score = auto_should_use_rag(prompt)
        route_note = f"\n\n> Routing: {'Docs' if use_rag else 'General'} (confidence {score:.2f})"

    # Generate response
    with st.spinner("Thinking…"):
        if use_rag:
            rag_chain = get_rag_chain()
            if rag_chain is None:
                answer = "I don’t have any documents indexed yet. Upload a PDF or switch to General chat."
                sources = []
            else:
                result = rag_chain.invoke({"question": prompt})
                answer = result.get("answer") or result.get("result") or ""
                sources = result.get("source_documents", [])
            # Track both histories
            st.session_state.general_history.append(HumanMessage(content=prompt))
            st.session_state.general_history.append(AIMessage(content=answer))
        else:
            answer = general_chat(prompt)
            sources = []

            # Update general chat memory
            st.session_state.general_history.append(HumanMessage(content=prompt))
            st.session_state.general_history.append(AIMessage(content=answer))

    # Render assistant
    final_text = answer + (route_note if route_note else "")
    st.chat_message("assistant").markdown(final_text)
    st.session_state.messages.append({"role": "assistant", "content": final_text})

    # Citations
    if show_citations and sources:
        with st.expander("Sources"):
            for i, doc in enumerate(sources, start=1):
                src = doc.metadata.get("source", "Unknown.pdf")
                page = doc.metadata.get("page", "N/A")
                st.markdown(f"- {i}. **{os.path.basename(src)}**, page {page}")
