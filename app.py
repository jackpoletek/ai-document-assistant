from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama

# Folder where the documents are stored
DOCUMENTS_DIR = Path("documents")

FAISS_INDEX_DIR = "faiss_index"


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


def create_embeddings() -> HuggingFaceEmbeddings:
    """Creates a local embedding model using HuggingFaceEmbeddings."""

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def create_vectorstore(
        chunks: list[Document],
        embeddings: HuggingFaceEmbeddings,
) -> FAISS:
    """Loads an existing FAISS index or creates a new one."""

    index_path = Path(FAISS_INDEX_DIR)

    if index_path.exists():
        print("\nLoading existing FAISS index...")

        return FAISS.load_local(
            folder_path=FAISS_INDEX_DIR,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )

    print("\nCreating new FAISS index...")

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    vector_store.save_local(FAISS_INDEX_DIR)

    print("FAISS index saved")

    return vector_store


def search_documents(
        vector_store: FAISS,
        question: str
) -> list[Document]:
    """Searches for document chunks similar to the question."""

    return vector_store.similarity_search(
        question,
        k=3
    )


def create_llm() -> ChatOllama:
    """Creates a local LLM using ChatOllama."""

    return ChatOllama(
        model="llama3.2:3b",
        temperature=0,
    )


def answer_question(
        llm: ChatOllama,
        question: str,
        documents: list[Document]
) -> str:
    """Generates an answer using the retrieved document chunks."""

    context = "\n\n".join(
        document.page_content for document in documents
    )

    prompt = f"""
    You are a helpful AI assistant.

    Use ONLY the information provided in the context.

    If the answer cannot be found in the context, say exactly this:

    "I couldn't find that information in the documents provided."

    Do not invent information.

    Context: {context}

    Question: {question}

    Answer:

    """

    response = llm.invoke(prompt)

    return response.content


def main():
    # Load documents and split them into chunks
    documents = load_documents()
    chunks = split_documents(documents)

    # Create embeddings and vector database
    embeddings = create_embeddings()
    vector_store = create_vectorstore(
        chunks,
        embeddings
    )

    # Create local LLM
    llm = create_llm()

    print("\nAI Document Assistant")
    print("-" * 60)
    print(f"Loaded documents : {len(documents)}")
    print(f"Created chunks : {len(chunks)}")
    print(f"Stored vectors : {vector_store.index.ntotal}")

    while True:
        question = input("\nAsk a question (or type 'exit'): ")

        if question.lower() == "exit":
            print("Goodbye!")
            break

        results = search_documents(
            vector_store,
            question,
        )

        answer = answer_question(
            llm,
            question,
            results
        )

        print("\nAnswer")
        print("-" * 60)
        print(answer)

        print("\nSources")
        print("-" * 60)

        shown = set()

        for document in results:
            source = document.metadata["source"]

            if source not in shown:
                print(f"- {source}")
                shown.add(source)


if __name__ == "__main__":
    main()
