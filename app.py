from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama

# Folder where the documents are stored
DOCUMENTS_DIR = Path("documents")


def load_documents():
    """Loads all .txt files from the documents directory and prints their names and contents."""
    documents = []

    for file_path in DOCUMENTS_DIR.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": file_path.name
                }
            )
        )

    return documents


def split_documents(documents):
    """Splits the documents into smaller chunks using RecursiveCharacterTextSplitter."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=120,
        chunk_overlap=30,
    )
    return splitter.split_documents(documents)


def create_embeddings():
    """Creates a local embedding model using HuggingFaceEmbeddings."""

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def create_vectorstore(chunks, embeddings):
    """Creates a FAISS vector store from the embeddings and chunks."""

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vector_store


def search_documents(vector_store, question):
    """Searches for document chunks similar to the question."""

    results = vector_store.similarity_search(
        question,
        k=3
    )

    return results


def create_llm():
    """Creates a local LLM using ChatOllama."""

    return ChatOllama(
        model="llama3.2:3b",
        temperature=0,
    )


def answer_question(llm, question, documents):
    """Generates an answer using the retrieved document chunks."""

    context = "\n\n".join(
        document.page_content for document in documents
    )

    prompt = f"""
    You are a helpful assistant.
    Answer the following question based on the context below using ONLY the information provided.
    If the answer is not contained within the context, respond with "I don't know."

    Context: {context}

    Question: {question}

    Answer:

    """

    response = llm.invoke(prompt)

    return response.content


def main():
    documents = load_documents()

    chunks = split_documents(documents)

    embeddings = create_embeddings()

    vector_store = create_vectorstore(
        chunks,
        embeddings
    )

    print(f"Loaded documents: {len(documents)}")
    print(f"Created chunks: {len(chunks)}")
    print(f"Vector store contains: {vector_store.index.ntotal} vectors.")
    print()

    question = input("Enter your question: ")

    results = search_documents(
        vector_store,
        question
    )

    llm = create_llm()

    answer = answer_question(
        llm,
        question,
        results
    )

    print("\nAnswer:")
    print("=" * 60)
    print(answer)


if __name__ == "__main__":
    main()
