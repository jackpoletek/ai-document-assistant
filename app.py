from pathlib import Path

# Folder where the documents are stored
DOCUMENTS_DIR = Path("documents")


def load_documents():
    """Loads all .txt files from the documents directory and prints their names and contents."""
    documents = []

    for file_path in DOCUMENTS_DIR.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")

        documents.append({
            "filename": file_path.name,
            "content": text
        })

    return documents


def main():
    documents = load_documents()

    for document in documents:
        print("=" * 50)
        print(document["filename"])
        print("-" * 50)
        print(document["content"])
        print()


if __name__ == "__main__":
    main()
