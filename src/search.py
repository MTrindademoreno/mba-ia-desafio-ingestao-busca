import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_postgres import PGVector

from ingest import (
    _connection_string_for_pgvector,
    _get_embedding_and_dim,
    _pg_vector_collection_name,
)

load_dotenv()

# Prompt após CONTEXTO (resultados concatenados do banco de dados).
PROMPT_APOS_CONTEXTO = """REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta do usuário}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

K = 10  # número de chunks mais relevantes na busca


@lru_cache(maxsize=1)
def get_vector_store() -> PGVector:
    """Mesma conexão, collection e embeddings que o ingest."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL não configurado no .env.")
    connection_url = _connection_string_for_pgvector(database_url)
    embedding, vector_size = _get_embedding_and_dim()
    collection_name = _pg_vector_collection_name()
    return PGVector(
        connection=connection_url,
        collection_name=collection_name,
        embeddings=embedding,
        embedding_length=vector_size,
        use_jsonb=True,
    )


def search_prompt(pergunta: str) -> str:
    """
    Busca os K chunks mais relevantes para a pergunta e monta o prompt.
    Retorna o prompt pronto para ser enviado ao LLM.
    """
    store = get_vector_store()
    docs = store.similarity_search(pergunta, k=K)
    resultados_concatenados = "\n\n".join(doc.page_content for doc in docs)
    corpo = PROMPT_APOS_CONTEXTO.replace("{pergunta do usuário}", pergunta)
    return f"CONTEXTO:\n{resultados_concatenados}\n\n{corpo}"
