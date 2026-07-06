from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

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


def main():
    documents = load_documents()

    chunks = split_documents(documents)

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
