"""
Interactive Full-Stack AI Chatbot (Streamlit + LangChain + Gemini)

Features
--------
* Streaming token-by-token output  (st.write_stream)
* Chat UI with persistent history  (st.chat_message / st.chat_input)
* Clear session-state management   (st.session_state)
* Conversation memory              (sliding window passed to the LLM)
* RAG over uploaded PDF/TXT/MD     (FAISS + local HuggingFace embeddings)
* History-aware retrieval          (follow-up questions are rewritten to standalone ones)
* Source citations, chat export, and clear/reset controls
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --------------------------------------------------------------------------- #
# Page configuration
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="AI Chatbot | RAG + Streaming",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

DEFAULT_MODEL = "gemini-3.8-flash"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
WELCOME = (
    "👋 Hi! I'm your AI assistant. Upload PDF/TXT/MD files in the sidebar and "
    "click **Build knowledge base** to chat with your documents — or just ask "
    "me anything."
)

# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given the chat history and the latest user question, rewrite the "
            "question so it is fully standalone (resolve pronouns and "
            "references). Do NOT answer it. Return only the rewritten question.",
        ),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful, precise AI assistant.\n"
            "Use the CONTEXT below to answer. If the answer is not in the "
            "context, say so clearly, then offer what general knowledge you can.\n"
            "Be concise, use markdown where helpful.\n\n"
            "CONTEXT:\n{context}",
        ),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)

CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful, friendly and precise AI assistant. "
            "Be concise and use markdown where helpful.",
        ),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)


# --------------------------------------------------------------------------- #
# Session-state management
# --------------------------------------------------------------------------- #
def init_state() -> None:
    """Create every session_state key exactly once."""
    defaults = {
        "messages": [{"role": "assistant", "content": WELCOME, "sources": []}],
        "vectorstore": None,
        "kb_files": [],      # names of indexed files
        "kb_chunks": 0,      # number of indexed chunks
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_chat() -> None:
    st.session_state.messages = [{"role": "assistant", "content": WELCOME, "sources": []}]


def reset_knowledge_base() -> None:
    st.session_state.vectorstore = None
    st.session_state.kb_files = []
    st.session_state.kb_chunks = 0


def to_lc_history(messages: list[dict], window: int) -> list[BaseMessage]:
    """Convert stored messages to LangChain messages (sliding memory window)."""
    # drop the welcome message, keep the last `window` exchanges
    convo = [m for m in messages if m["content"] != WELCOME]
    convo = convo[-(window * 2):] if window > 0 else []
    return [
        HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
        for m in convo
    ]


def transcript_markdown(messages: list[dict]) -> str:
    lines = [f"# Chat transcript — {datetime.now():%Y-%m-%d %H:%M}\n"]
    for m in messages:
        who = "**You**" if m["role"] == "user" else "**Assistant**"
        lines.append(f"{who}: {m['content']}\n")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading embedding model (first run only)…")
def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


@st.cache_resource(show_spinner=False)
def get_llm(api_key: str, model: str, temperature: float):
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=api_key,
    )


# --------------------------------------------------------------------------- #
# RAG helpers
# --------------------------------------------------------------------------- #
def build_knowledge_base(uploaded_files) -> tuple[int, list[str]]:
    """Load, split, embed and index the uploaded files into FAISS."""
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    docs, names = [], []
    for f in uploaded_files:
        suffix = Path(f.name).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(f.getbuffer())
            tmp_path = tmp.name
        try:
            loader = PyPDFLoader(tmp_path) if suffix == ".pdf" else TextLoader(tmp_path, encoding="utf-8")
            loaded = loader.load()
            for d in loaded:
                d.metadata["source"] = f.name  # show the real filename, not the temp path
            docs.extend(loaded)
            names.append(f.name)
        finally:
            os.unlink(tmp_path)

    if not docs:
        raise ValueError("No readable text found in the uploaded files.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    if not chunks:
        raise ValueError("Documents were empty after splitting.")

    st.session_state.vectorstore = FAISS.from_documents(chunks, get_embeddings())
    return len(chunks), names


def retrieve(question: str, history: list[BaseMessage], llm, k: int):
    """History-aware retrieval. Returns (context_text, sources, standalone_question)."""
    standalone = question
    if history:
        rewrite = REWRITE_PROMPT | llm | StrOutputParser()
        try:
            standalone = rewrite.invoke({"history": history, "question": question}).strip() or question
        except Exception:
            standalone = question

    results = st.session_state.vectorstore.similarity_search(standalone, k=k)
    context = "\n\n---\n\n".join(d.page_content for d in results)
    sources = [
        {
            "source": d.metadata.get("source", "unknown"),
            "page": (d.metadata.get("page") + 1) if isinstance(d.metadata.get("page"), int) else None,
            "snippet": d.page_content[:300].replace("\n", " ") + "…",
        }
        for d in results
    ]
    return context, sources, standalone


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
init_state()

st.title("🤖 AI Chatbot")
st.caption("Streaming • Conversation memory • RAG over your documents • Powered by LangChain + Gemini")

# ---- Sidebar --------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Settings")

    api_key = st.text_input(
        "Google API key",
        value=os.environ.get("GOOGLE_API_KEY", ""),
        type="password",
        help="Free key: https://aistudio.google.com/apikey",
    )
    model = st.text_input("Model", value=DEFAULT_MODEL)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)
    memory_window = st.slider("Memory (past exchanges)", 0, 20, 6, help="How many previous Q&A pairs the model remembers.")
    top_k = st.slider("Retrieved chunks (k)", 1, 8, 4)

    st.divider()
    st.header("📚 Knowledge base")
    files = st.file_uploader(
        "Upload PDF / TXT / MD",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )
    c1, c2 = st.columns(2)
    if c1.button("Build", type="primary", use_container_width=True, disabled=not files):
        with st.spinner("Indexing documents…"):
            try:
                n_chunks, names = build_knowledge_base(files)
                st.session_state.kb_chunks, st.session_state.kb_files = n_chunks, names
                st.success(f"Indexed {n_chunks} chunks from {len(names)} file(s).")
            except Exception as e:
                st.error(f"Indexing failed: {e}")
    if c2.button("Remove", use_container_width=True, disabled=st.session_state.vectorstore is None):
        reset_knowledge_base()
        st.rerun()

    if st.session_state.vectorstore is not None:
        st.info(f"RAG **ON** — {st.session_state.kb_chunks} chunks\n\n" + "\n".join(f"• {n}" for n in st.session_state.kb_files))
    else:
        st.caption("RAG **OFF** — answering from general knowledge.")

    st.divider()
    st.header("💬 Session")
    n_turns = sum(1 for m in st.session_state.messages if m["role"] == "user")
    st.metric("Questions asked", n_turns)
    if st.button("🗑️ Clear chat", use_container_width=True):
        reset_chat()
        st.rerun()
    st.download_button(
        "⬇️ Export chat (.md)",
        data=transcript_markdown(st.session_state.messages),
        file_name="chat_transcript.md",
        mime="text/markdown",
        use_container_width=True,
    )

# ---- Render history -------------------------------------------------------- #
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📄 Sources ({len(msg['sources'])})"):
                for s in msg["sources"]:
                    page = f" — p.{s['page']}" if s["page"] else ""
                    st.markdown(f"**{s['source']}**{page}")
                    st.caption(s["snippet"])

# ---- Handle new input ------------------------------------------------------ #
if prompt := st.chat_input("Ask me anything…"):
    if not api_key:
        st.error("Please enter your Google API key in the sidebar.")
        st.stop()

    # memory snapshot BEFORE adding the new question
    history = to_lc_history(st.session_state.messages, memory_window)

    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        try:
            llm = get_llm(api_key, model, temperature)

            sources: list[dict] = []
            if st.session_state.vectorstore is not None:
                with st.spinner("Searching documents…"):
                    context, sources, _ = retrieve(prompt, history, llm, top_k)
                chain = RAG_PROMPT | llm | StrOutputParser()
                inputs = {"context": context, "history": history, "question": prompt}
            else:
                chain = CHAT_PROMPT | llm | StrOutputParser()
                inputs = {"history": history, "question": prompt}

            # Real-time token streaming
            answer = st.write_stream(chain.stream(inputs))

            if sources:
                with st.expander(f"📄 Sources ({len(sources)})"):
                    for s in sources:
                        page = f" — p.{s['page']}" if s["page"] else ""
                        st.markdown(f"**{s['source']}**{page}")
                        st.caption(s["snippet"])

            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
        except Exception as e:
            msg = str(e)
            if "API key" in msg or "API_KEY" in msg or "403" in msg or "401" in msg:
                st.error("Authentication failed — please check your Google API key.")
            elif "429" in msg or "quota" in msg.lower():
                st.error("Rate limit / quota reached. Wait a moment and try again.")
            else:
                st.error(f"Something went wrong: {msg}")
            # keep state consistent: remove the unanswered user turn
            st.session_state.messages.pop()
