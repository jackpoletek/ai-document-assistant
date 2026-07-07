from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings

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


def main():
    documents = load_documents()

    chunks = split_documents(documents)

    embeddings = create_embeddings(chunks)

    vectors = embeddings.embed_documents(
        [chunk.page_content for chunk in chunks]
    )

    print(f"Created {len(vectors)} embeddings.")
    print()

    print("First embedding (first 10 numbers):")
    print(vectors[0][:10])

    print(f"Loaded documents: {len(documents)}")
    print(f"Created chunks: {len(chunks)}")
    print()

    for i, chunk in enumerate(chunks, start=1):
        print("=" * 60)
        print(f"Chunk {i}")
        print(f"Source: {chunk.metadata['source']}")
        print("-" * 60)
        print(chunk.page_content)
        print()


if __name__ == "__main__":
    main()
