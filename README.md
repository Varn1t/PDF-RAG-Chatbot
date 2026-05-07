<div align="center">

# 📚 RAG Chatbot

**Chat with any PDF or YouTube video — 100% locally, zero API costs.**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain.com)
[![Ollama](https://img.shields.io/badge/Ollama-llama3-black?style=for-the-badge)](https://ollama.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

</div>

---
<img width="1253" height="606" alt="image" src="https://github.com/user-attachments/assets/d0f78737-6492-4f34-8f08-b15b902681d4" />

## 🧠 What is RAG?

Large Language Models (LLMs) are powerful, but they don't know the contents of *your* documents. **Retrieval-Augmented Generation (RAG)** solves this by combining a vector search engine with an LLM:

```
Your PDF / YouTube Video
   │
   ▼
[Chunking]  →  Split into smaller, overlapping pieces
   │
   ▼
[Embedding]  →  Convert each chunk into a vector (numerical meaning)
   │
   ▼
[FAISS Index]  →  Store vectors for fast similarity search
   │
   ▼
[Your Question]  →  Find the most relevant chunks
   │
   ▼
[LLM + Context]  →  Generate a grounded, accurate answer ✅
```

---

## ✨ Features

| Feature | Description |
|---|---|
| 📄 PDF Upload | Drag and drop any PDF directly in the browser |
| 🎥 YouTube Support | Paste any YouTube URL to chat with video transcripts |
| 🖥️ CLI Mode | Run the full pipeline from the terminal with `main.py` |
| 🔒 Fully Local | No OpenAI key, no API costs — runs entirely on your machine |
| 🎯 Grounded Answers | Responses are based on your document, not hallucinated |
| 💬 Chat History | Conversation persists throughout your session |
| ⚡ Auto Re-embed | Switch sources and it re-indexes automatically |

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| 🤖 LLM | `llama3` via [Ollama](https://ollama.com) |
| 🔢 Embeddings | `all-MiniLM-L6-v2` (HuggingFace Sentence Transformers) |
| 🗄️ Vector Store | FAISS (CPU) |
| 📑 PDF Parsing | LangChain + PyPDF |
| 🎥 YouTube Transcripts | `youtube-transcript-api` |
| 🖥️ Web UI | Streamlit |

---

## 🚀 Setup

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com) installed and running
- ~5 GB disk space for the llama3 model

### Step-by-step

**1. Clone the repository**
```bash
git clone https://github.com/Varn1t/PDF-RAG-Chatbot.git
cd PDF-RAG-Chatbot
```

**2. Install Python dependencies**
```bash
pip install -r requirements.txt
```

**3. Pull the LLM model with Ollama**
```bash
ollama pull llama3
```

**4a. Launch the Streamlit web app**
```bash
streamlit run app.py
```
Then open **http://localhost:8501** in your browser.

**4b. Or run the CLI version**
```bash
python main.py
```

---

## 🗂️ Project Structure

```
PDF-RAG-Chatbot/
├── app.py              # Streamlit web app (PDF + YouTube)
├── main.py             # Interactive CLI version (PDF + YouTube)
├── requirements.txt    # Python dependencies
├── .gitignore
└── README.md
```

---

## ⚙️ How It Works

### Web App (`app.py`)
1. **Upload** a PDF or paste a YouTube URL in the Streamlit UI
2. The app **chunks** the content into 1,000-character segments with 50-character overlap
3. Each chunk is **embedded** using `all-MiniLM-L6-v2` and stored in a FAISS index
4. When you ask a question, the **top 3 most relevant chunks** are retrieved
5. Those chunks are passed as context to **llama3**, which generates a precise answer

### CLI (`main.py`)
1. Choose to load a **PDF** (local path or URL) or **YouTube video** (URL)
2. The script builds the same RAG pipeline in your terminal
3. Enter your questions in an interactive chat loop — type `exit` to quit

---

<div align="center">

Made by Varnit using LangChain, FAISS, Ollama, Streamlit, and youtube-transcript-api

</div>
