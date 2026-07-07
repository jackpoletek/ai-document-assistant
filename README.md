# AI Document Assistant

A simple Retrieval-Augmented Generation (RAG) application built with Python, LangChain, FAISS, local Hugging Face embeddings and Ollama.

Instead of relying only on a language model's built-in knowledge, the application searches your own documents first, retrieves the most relevant information, and then uses a local LLM to generate an answer.

---

## Features

- Load text documents from a local folder
- Split documents into smaller chunks
- Generate semantic embeddings
- Store embeddings in a FAISS vector database
- Persist the FAISS index between runs
- Retrieve the most relevant document chunks
- Generate answers using a local Ollama model
- Display the source documents used to answer the question

---

## Technologies

- Python 3.12.8
- LangChain
- LangChain HuggingFace
- Sentence Transformers
- FAISS
- Ollama
- Llama 3.2:3B
- Git
- VS Code

---

## Project Structure

```
ai-document-assistant/
│
├── documents/
│ ├── django_auth.txt
│ ├── stripe.txt
│ ├── models.txt
│ ├── views.txt
│ ├── payments.txt
│ └── forms.txt
│
├── faiss_index/
├── venv/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd ai-document-assistant
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

### Windows

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Install Ollama:

https://ollama.com/download

Download the language model:

```bash
ollama pull llama3.2:3b
```

---

## Running the application

```bash
python app.py
```

Example:

```
Ask a question:

How do Stripe webhooks work?
```

Example output:

```
Answer

Stripe uses webhooks to notify the application when a payment succeeds or fails.
Webhook signatures should always be verified before processing the event.

Sources

- stripe.txt
- payments.txt
```

---

## How RAG Works

```
Documents
│
v
Load documents
│
v
Split into chunks
│
v
Generate embeddings
│
v
Store in FAISS
│
v
User asks a question
│
v
Retrieve relevant chunks
│
v
Send context to Ollama
│
v
Generate answer
```

---

## Example Questions

- How do Stripe webhooks work?
- How does Django authentication work?
- What is a Django model?
- How are payments processed?
- What are Django views responsible for?

---

## Future Improvements

- Support PDF documents
- Support multiple document formats
- Web interface using Django or Flask
- Conversation memory
- Streaming responses
- Docker deployment
- Unit tests
- Better prompt engineering

---

## What I Learned

This project helped me understand the core concepts behind Retrieval-Augmented Generation (RAG), including:

- Document chunking
- Embeddings
- Semantic search
- Vector databases
- Retrieval pipelines
- Local LLMs
- Prompt engineering
- Reducing hallucinations using retrieved context