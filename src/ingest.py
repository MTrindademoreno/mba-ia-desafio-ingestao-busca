import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import RunnableLambda
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Dimensões padrão (pgvector exige que coincida com o vetor da API)
# gemini-embedding-001 → 3072; models/embedding-001 (legacy) → 768
EMBEDDING_DIM_GOOGLE_DEFAULT = 3072
EMBEDDING_DIM_OPENAI = 1536  # text-embedding-3-small

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def _pg_vector_collection_name() -> str:
    name = os.getenv("PG_VECTOR_COLLECTION_NAME", "").strip()
    if not name:
        raise ValueError(
            "PG_VECTOR_COLLECTION_NAME é obrigatório no .env "
            "(nome da collection no vector store)."
        )
    return name


def _connection_string_for_pgvector(url: str) -> str:
    """Garante que a URL use o driver psycopg para langchain-postgres."""
    if not url:
        return url
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url.split("://", 1)[1]
    if url.startswith("postgresql+psycopg://"):
        return url
    return url


def _google_embedding_dimension() -> int:
    raw = os.getenv("GOOGLE_EMBEDDING_DIM", "").strip()
    if raw:
        return int(raw)
    return EMBEDDING_DIM_GOOGLE_DEFAULT


def _get_embedding_and_dim():
    """Retorna (embedding instance, vector_size) conforme as chaves de API configuradas."""
    google_key = os.getenv("GOOGLE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    if google_key:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        dim = _google_embedding_dimension()
        return (
            GoogleGenerativeAIEmbeddings(
                model=os.getenv(
                    "GOOGLE_EMBEDDING_MODEL", "gemini-embedding-001"
                )
            ),
            dim,
        )
    if openai_key:
        from langchain_openai import OpenAIEmbeddings
        return (
            OpenAIEmbeddings(
                model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
            ),
            EMBEDDING_DIM_OPENAI,
        )
    raise ValueError(
        "Defina GOOGLE_API_KEY ou OPENAI_API_KEY no .env para gerar embeddings."
    )


def _build_load_split_chain(pdf_path: str):
    """
    LCEL: carregar PDF -> dividir em chunks (RunnableSequence).
    O splitter desta versão não é Runnable; encapsulamos com RunnableLambda.
    """
    loader = PyPDFLoader(pdf_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    return RunnableLambda(lambda _: loader.load()) | RunnableLambda(
        lambda docs: splitter.split_documents(docs)
    )


def ingest_pdf():
    pdf_path = os.getenv("PDF_PATH")
    if not pdf_path or not Path(pdf_path).exists():
        raise FileNotFoundError(
            f"PDF_PATH não configurado ou arquivo não existe: {pdf_path}"
        )
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL não configurado no .env.")

    connection_url = _connection_string_for_pgvector(database_url)
    embedding, vector_size = _get_embedding_and_dim()

    # 1–2) Chain LCEL: carregar PDF -> split em chunks
    load_split = _build_load_split_chain(pdf_path)
    chunks = load_split.invoke(None)

    if not chunks:
        print("Nenhum conteúdo extraído do PDF.")
        return

    pages_meta = {c.metadata.get("page", 0) for c in chunks}
    num_pages = len(pages_meta)
    print(f"PDF carregado: {num_pages} página(s) -> {len(chunks)} chunk(s).")

    collection_name = _pg_vector_collection_name()

    # 3) Embeddings + persistência no PGVector
    PGVector.from_documents(
        documents=chunks,
        embedding=embedding,
        connection=connection_url,
        collection_name=collection_name,
        embedding_length=vector_size,
        use_jsonb=True,
    )

    print(f"Vetores armazenados na collection '{collection_name}'.")


if __name__ == "__main__":
    ingest_pdf()