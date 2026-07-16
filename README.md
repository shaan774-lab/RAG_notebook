# 📚 Smart PDF Gemini RAG Notebook

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg?style=for-the-badge&logo=streamlit)](https://streamlit.io/)
[![FAISS](https://img.shields.io/badge/VectorDB-FAISS-green.svg?style=for-the-badge)](https://github.com/facebookresearch/faiss)
[![Gemini](https://img.shields.io/badge/LLM-Gemini%20Pro-4285F4.svg?style=for-the-badge&logo=google)](https://deepmind.google/technologies/gemini/)

An interactive Retrieval-Augmented Generation (RAG) notebook that allows users to upload a PDF, parse it, index its contents semantically into a local vector database, and perform context-aware questioning, summarization, and mind-map generation using the Google Gemini API.

---

## 📊 Technical Flow Diagram

The diagram below details the end-to-end workflow from PDF upload and ingestion to FAISS vector search and Gemini-powered generation:

```mermaid
graph TD
    %% Ingestion Flow
    subgraph IP [Ingestion Pipeline]
        A[User Uploads PDF] --> B[Save PDF to local /uploads]
        B --> C[Extract Text using PyPDF2]
        C --> D[Chunk Text: RecursiveCharacterSplitter size=500, overlap=100]
        D --> E[Generate Embeddings: all-MiniLM-L6-v2]
        E --> F[Store in FAISS Vector DB: embeddings.faiss & metadata.pkl]
    end

    %% Query and RAG Flow
    subgraph RAG [Retrieval & Generation RAG]
        G[User enters Query/Prompt] --> H[Generate Query Embedding]
        F -->|Search Vector Space| I[FAISS Semantic Search: Retrieve top-k chunks]
        H --> I
        I --> J[Construct Context-augmented Prompt]
        J --> K[Gemini API: gemini-pro]
        K --> L{User Action}
        L -->|1. Ask Question| M[Contextual QA Answer]
        L -->|2. Generate Summary| N[Bullet Point Summary]
        L -->|3. Generate Mindmap| O[Hierarchical Text Mind Map]
    end

    style IP fill:#111827,stroke:#3b82f6,stroke-width:2px,color:#fff
    style RAG fill:#111827,stroke:#10b981,stroke-width:2px,color:#fff
```

---

## ⚡ Key Features

* **PDF Ingestion:** Automatic text extraction from PDF files using `PyPDF2`.
* **Semantic Chunking:** Custom splitting with overlap configured to keep context across chunk boundaries.
* **Local Vector Store:** Utilizes **FAISS** (Facebook AI Similarity Search) and `sentence-transformers` (`all-MiniLM-L6-v2`) to produce and index 384-dimensional vector embeddings locally.
* **Retrieval-Augmented Answering:** Queries the local vector store for context matches and appends them to prompts for ground truth generation.
* **Multi-Format AI Output:**
  - **QA:** Solves complex user questions based only on retrieved context.
  - **Summary:** Generates short, bulleted summaries of retrieved content.
  - **Mind-maps:** Automatically creates structured hierarchical text representations of the selected topic.

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **RAG Orchestrator:** LangChain
* **Vector Store:** FAISS (cpu)
* **Embedding Model:** SentenceTransformers (`all-MiniLM-L6-v2`)
* **Generative API:** Google Generative AI (Gemini Pro)
* **Document Parsing:** PyPDF2

---

## 🚀 Getting Started

Follow these steps to run the application locally:

### Step 1: Clone the Repository
```bash
git clone https://github.com/shaan774-lab/RAG_notebook.git
cd RAG_notebook
```

### Step 2: Set Up a Virtual Environment

**Using venv:**
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

**Using Conda:**
```bash
conda create -n rag_env python=3.8 -y
conda activate rag_env
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure API Keys
Create a `.env` file in the root directory and add your Google Gemini API Key:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### Step 5: Start the Streamlit App
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📂 Project Structure

```text
RAG_notebook/
│
├── backend/
│   ├── vector_store/           # Directory where FAISS indices are saved
│   ├── geminiapi.py            # Interfaces with Google Generative AI API
│   └── rag_engine.py           # Embeddings, chunking, and FAISS indexing
│
├── uploads/                    # Local PDF uploads folder
├── app.py                      # Main Streamlit frontend interface
├── requirements.txt            # Python dependencies
└── .gitignore                  # Git exclusions (ignores vectors, env variables)
```

---

## 👥 Contributors

* **Shaan Saxena** - [shaan774-lab](https://github.com/shaan774-lab)
