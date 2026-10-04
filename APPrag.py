import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# Load credentials from .env file
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Page setup
st.set_page_config(page_title="RAG Chatbot with Memory", page_icon="📄")
st.title("📄 RAG Chatbot with Memory (ChromaDB)")

# Sidebar
with st.sidebar:
    st.header("Settings")
    if not groq_api_key:
        groq_api_key = st.text_input("Enter Groq API Key", type="password")
    session_id = st.text_input("Session ID", value="default_session")
    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
    if st.button("Clear Chat History"):
        st.session_state.clear()
        st.rerun()

if not groq_api_key:
    st.warning("Please enter your Groq API key in the sidebar to continue.")
    st.stop()

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model="openai/gpt-oss-20b")

# Initialize embeddings (free, runs locally, small model)
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = get_embeddings()

# Build / load the ChromaDB vectorstore from the uploaded file
def build_vectorstore(file):
    suffix = os.path.splitext(file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path) if suffix == ".pdf" else TextLoader(tmp_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return vectorstore

if uploaded_file is not None and "vectorstore" not in st.session_state:
    with st.spinner("Processing document and storing embeddings in ChromaDB..."):
        st.session_state.vectorstore = build_vectorstore(uploaded_file)
    st.success("Document indexed. You can now ask questions about it.")

retriever = None
if "vectorstore" in st.session_state:
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})

# Prompt template: system + retrieved context + conversation history
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant. Use the context below to answer the question "
     "if it is relevant. If the context does not contain the answer, answer "
     "using your own knowledge.\n\nContext:\n{context}"),
    MessagesPlaceholder(variable_name="messages")
])

def format_context(input_dict):
    question = input_dict["messages"][-1].content
    if retriever is None:
        return ""
    docs = retriever.invoke(question)
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    RunnablePassthrough.assign(context=format_context)
    | prompt
    | llm
)

# Session-based message store
if "store" not in st.session_state:
    st.session_state.store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages"
)

# Render existing conversation history
history = get_session_history(session_id)
for msg in history.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# Chat input box
user_input = st.chat_input("Type your message...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    config = {"configurable": {"session_id": session_id}}
    response = with_message_history.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config
    )

    with st.chat_message("assistant"):
        st.markdown(response.content)