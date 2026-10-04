# 📄 RAG Chatbot with Conversation Memory

A single-file Streamlit web app that combines **Retrieval-Augmented 
Generation (RAG)** with **persistent chat memory**, letting you upload 
a document and have a real, context-aware conversation about its contents.

## Features

- 🧠 **Conversation Memory** — Remembers the full chat history per session, 
  so follow-up questions work naturally (powered by LangChain's 
  `RunnableWithMessageHistory`)
- 📚 **Document-Aware Answers (RAG)** — Upload a PDF or TXT file; the app 
  chunks it, embeds it, and stores it in **ChromaDB** for semantic search
- ⚡ **Fast, Free LLM** — Uses Groq's free-tier API (OpenAI GPT-OSS models) 
  for near-instant responses — no OpenAI key required
- 🔒 **Local Embeddings** — Uses Hugging Face's `all-MiniLM-L6-v2` model, 
  running locally and free, with no extra API cost
- 💬 **Clean Chat UI** — Built with Streamlit's native chat components

## Tech Stack

- **LangChain** — orchestration (chains, retrievers, memory)
- **Streamlit** — web UI
- **ChromaDB** — vector database for document storage/search
- **Groq API** — LLM inference (free tier)
- **Hugging Face Transformers** — text embeddings

## How It Works

1. Upload a PDF or TXT document
2. The document is split into chunks and converted into embeddings
3. Embeddings are stored in a local ChromaDB vector store
4. When you ask a question, the retriever finds the most relevant chunks
5. Those chunks + your full conversation history are sent to the LLM
6. The LLM answers using the document content, while remembering context 
   from earlier in the chat

## Setup

\`\`\`bash
pip install -r requirements.txt
\`\`\`

Create a `.env` file:
\`\`\`
GROQ_API_KEY=your_key_here
\`\`\`

Get a free key at [console.groq.com/keys](https://console.groq.com/keys)

## Run

\`\`\`bash
streamlit run app_rag.py
\`\`\`
