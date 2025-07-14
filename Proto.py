import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain_groq import ChatGroq

# Set your Groq API key
os.environ["GROQ_API_KEY"] = ""

# Load PDF and create index
@st.cache_resource
def load_pdf(pdf_name):
    loaders = [PyPDFLoader(pdf_name)]
    index = VectorstoreIndexCreator(
        vectorstore_cls=Chroma,
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200),
        embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2")
    ).from_loaders(loaders)
    return index

# Initialize LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.5,
    max_tokens=200
)

# UI
col1, col2 = st.columns([1, 5])
with col1:
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)  # Adds space above image
    st.image("bot.png", width=60)
with col2:
    st.markdown(
        "<span style='font-size: 2.5em; font-weight: bold; margin-bottom: 20px; display: inline-block;'>BuddyBot - Your Office Buddy</span>",
        unsafe_allow_html=True
    )


pdf_name = "AnswersNewComer.pdf"
index = load_pdf(pdf_name)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=index.vectorstore.as_retriever(),
    input_key="question",
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

prompt = st.chat_input("Ask your questions")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Generating Response, Please wait..."):
        response = qa_chain.run(prompt)

    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
