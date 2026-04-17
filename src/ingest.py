import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.embeddings import Embeddings
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

# Lotes pequenos + pausa ajudam a evitar 429 na API Google (embeddings).
EMBED_BATCH_SIZE = 14
EMBED_BATCH_DELAY_SECONDS = 2
EMBED_429_MAX_RETRIES = 8


def _pg_vector_collection_name() -> str:
    name = os.getenv("PG_VECTOR_COLLECTION_NAME", "").strip()
    if not name:
        raise ValueError(
            "PG_VECTOR_COLLECTION_NAME é obrigatório no .env "
            "(nome da collection no vector store)."
        )
    return name


def _google_embed_content_with_retry(inner, req) -> list[float]:
    for attempt in range(EMBED_429_MAX_RETRIES):
        try:
            resp = inner.client.embed_content(req)
            return list(resp.embedding.values)
        except Exception as e:
            if attempt == EMBED_429_MAX_RETRIES - 1:
                raise
            msg = str(e).lower()
            if "429" in str(e) or "resource exhausted" in msg:
                time.sleep(min(90.0, 2.0**attempt))
            else:
                raise


class _SequentialGoogleEmbeddings(Embeddings):
    """
    Evita batch_embed_contents (429 comum no free tier). Usa embed_content por texto,
    com pausa e retry exponencial em ResourceExhausted/429.
    """

    def __init__(self, inner, delay_between_texts_s: float):
        self._inner = inner
        self._delay = delay_between_texts_s

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i, text in enumerate(texts):
            req = self._inner._prepare_request(text)
            out.append(_google_embed_content_with_retry(self._inner, req))
            if i < len(texts) - 1:
                time.sleep(self._delay)
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(text)


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
        inner = GoogleGenerativeAIEmbeddings(
            model=os.getenv("GOOGLE_EMBEDDING_MODEL", "gemini-embedding-001")
        )
        delay = float(os.getenv("EMBED_BATCH_DELAY_SECONDS", str(EMBED_BATCH_DELAY_SECONDS)))
        # Opt-in: padrão usa o SDK (batch_embed_contents). Ative se tiver 429 no free tier.
        use_sequential = os.getenv("GOOGLE_EMBED_SEQUENTIAL", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        embedding = (
            _SequentialGoogleEmbeddings(inner, delay_between_texts_s=delay)
            if use_sequential
            else inner
        )
        return (embedding, dim)
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
    batch_size = max(1, int(os.getenv("EMBED_BATCH_SIZE", str(EMBED_BATCH_SIZE))))
    delay_s = float(os.getenv("EMBED_BATCH_DELAY_SECONDS", str(EMBED_BATCH_DELAY_SECONDS)))

    batches = [chunks[i : i + batch_size] for i in range(0, len(chunks), batch_size)]
    store = None
    for i, batch in enumerate(batches):
        if store is None:
            store = PGVector.from_documents(
                documents=batch,
                embedding=embedding,
                connection=connection_url,
                collection_name=collection_name,
                embedding_length=vector_size,
                use_jsonb=True,
            )
        else:
            store.add_documents(batch)
        print(f"  Lote {i + 1}/{len(batches)} ({len(batch)} chunks).")
        if i < len(batches) - 1:
            time.sleep(delay_s)

    print(f"Vetores armazenados na collection '{collection_name}'.")


if __name__ == "__main__":
    ingest_pdf()