# Interactive Full-Stack AI Chatbot (Streamlit + LangChain)

Streaming chat UI with conversation memory and RAG over your own PDF/TXT/MD files.

## Features
- `st.chat_message` / `st.chat_input` chat UI
- Token streaming with `st.write_stream`
- `st.session_state` for history, knowledge base, and settings
- Sliding-window conversation memory
- RAG: FAISS + local HuggingFace embeddings, history-aware retrieval, source citations
- Clear chat, export transcript, error handling

## Run locally
```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
export GOOGLE_API_KEY=your_key                    # Windows: set GOOGLE_API_KEY=your_key
streamlit run app.py
```
Open http://localhost:8501. Free API key: https://aistudio.google.com/apikey

## Run in Google Colab
Open `Streamlit_AI_Chatbot.ipynb` and run all cells; a public link appears in the last cell.
