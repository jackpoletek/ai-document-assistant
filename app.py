from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

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


def main():
    documents = load_documents()

    chunks = split_documents(documents)

    embeddings = create_embeddings()

    vector_store = create_vectorstore(
        chunks,
        embeddings
    )

    question = input("Enter your question: ")

    results = search_documents(
        vector_store,
        question
    )

    print("\nMost relevant search results:")

    for i, result in enumerate(results, start=1):
        print("=" * 60)
        print(f"Result {i}")
        print(f"Source: {result.metadata['source']}")
        print("-" * 60)
        print(result.page_content)
        print()

    print(f"Loaded documents: {len(documents)}")
    print(f"Created chunks: {len(chunks)}")
    print(f"Vector store contains: {vector_store.index.ntotal} vectors.")
    print()


if __name__ == "__main__":
    main()
