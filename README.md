## 🩺 Medical Question Answering Using RAG

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white) ![Hugging Face](https://img.shields.io/badge/Embeddings-Hugging%20Face-FFD21E) ![ChromaDB](https://img.shields.io/badge/Vector%20DB-Chroma-7952B3) ![Groq](https://img.shields.io/badge/LLM-Groq-F55036)

An educational **Retrieval-Augmented Generation (RAG)** project that answers questions using passages retrieved from a medical PDF. Instead of asking a language model to answer from memory, the program retrieves relevant document chunks and includes them as context.

This project explores the complete path from **PDF text extraction** to **semantic search** and **LLM-generated answers with requested source references**.

## 🔎 RAG pipeline

1. **Load the PDF:** Extract text from `symptoms.pdf` using `pypdf`, preserving page numbers.
2. **Split the text:** Create overlapping chunks with `RecursiveCharacterTextSplitter` (110 tokens, 10-token overlap).
3. **Generate embeddings:** Encode chunks with `all-MiniLM-L6-v2` via Hugging Face.
4. **Store vectors:** Save embeddings and metadata locally with **ChromaDB** and cosine distance.
5. **Retrieve context:** Find the **three nearest chunks** for each user question.
6. **Generate an answer:** Pass the retrieved text and page references to `openai/gpt-oss-20b` through Groq.

## 🧩 Workflow

```text
symptoms.pdf
     |
     v
PDF extraction + page metadata
     |
     v
Text splitting --> Hugging Face embeddings
     |
     v
Chroma vector database
     ^
     |  semantic similarity search
User question
     |
     v
Top-3 retrieved chunks + source labels
     |
     v
Groq LLM --> Answer with requested [Source N] references
```

## 📂 Project files

```text
├── RAG.py             # PDF processing, embedding, retrieval, and Q&A
├── symptoms.pdf       # Example source document
├── Pipfile
└── Pipfile.lock
```

The program creates/uses a local `chroma/` directory for persistent vector data. It is **generated data** and generally does not need to be committed.

## ⚙️ Run locally

```bash
git clone https://github.com/YasiNoshirvan/Medical-Question-Answering-Using-RAG.git
cd Medical-Question-Answering-Using-RAG
python -m pip install pypdf langchain-core langchain-text-splitters langchain-huggingface langchain-chroma sentence-transformers openai python-dotenv tiktoken
```

Create a local `.env` file:

```dotenv
GROQ_API_KEY=your_groq_api_key
```

The current script imports `pysqlite3` and replaces the standard SQLite module as a **compatibility workaround for the original environment**. Install a compatible `pysqlite3` package for that environment, or remove the workaround if your newer Python/SQLite setup already meets Chroma's requirements.

Place `symptoms.pdf` next to `RAG.py`, and run:

```bash
python RAG.py
```

Enter a question in the terminal, or type `q` to quit.

## 💬 Example questions

```text
Which symptoms are discussed in the document?
Does the document describe possible causes of these symptoms?
What does the document say about diagnosis?
```

Answers depend on the supplied document; these examples are **not validated medical responses**.

## ⚠️ Limitations

- The system always retrieves the **three nearest chunks**; it does not yet apply a calibrated relevance threshold.
- It reprocesses the source PDF on each run, so indexing could be optimized.
- Source references are **requested in the prompt**, not independently verified against the full PDF.
- Medical accuracy has **not** been clinically evaluated. This is **not** a diagnostic or treatment tool.
- Only redistribute `symptoms.pdf` if its license permits sharing.

---

*An educational RAG pipeline focused on embeddings, vector retrieval, and document-grounded question answering.*
