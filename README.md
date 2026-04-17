# Desafio MBA — Engenharia de Software com IA (ingestão e busca)

Pipeline : PDF → chunks → embeddings → **Postgres + pgvector**; perguntas no terminal com busca por similaridade e LLM (**Google Gemini** ou **OpenAI**, conforme chaves no `.env`).

## Pré-requisitos

- Python 3.10+
- Docker e Docker Compose
- Chave **Google AI** e/ou **OpenAI** (embeddings e chat seguem a prioridade de chaves definida no código)

## 1. Subir o banco

```bash
docker compose up -d
```

Aguarde o Postgres ficar saudável. A porta padrão no host é **5432**; se estiver ocupada, ajuste o mapeamento no `docker-compose.yml` e o `DATABASE_URL` no `.env`.

## 2. Ambiente Python

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Configuração

```bash
cp .env.example .env
```

Preencha pelo menos:

- `DATABASE_URL` — ex.: `postgresql+psycopg://postgres:postgres@127.0.0.1:5432/rag` (alinhado ao `docker-compose`)
- `GOOGLE_API_KEY` **ou** `OPENAI_API_KEY`
- `PDF_PATH` — caminho do PDF a indexar
- `PG_VECTOR_COLLECTION_NAME` — **obrigatório** (nome lógico da collection no vector store; ex.: `meu_rag_collection`)

Comentários no `.env` devem começar com `#`.

## 4. Ingestão

```bash
python3 src/ingest.py
```

## 5. Chat

```bash
python3 src/chat.py
```

Digite perguntas; `sair` encerra.

## Troca de provedor ou modelo de embedding

Se você **mudar de Google para OpenAI** (ou o contrário) ou **trocar o modelo de embedding** (dimensão ou família diferente), **defina uma nova collection** no `.env`, alterando `PG_VECTOR_COLLECTION_NAME` para um nome que ainda não tenha sido usado com o modelo anterior, antes de rodar o ingest.

**Motivo:** embeddings de modelos diferentes têm **dimensões** e **espaços semânticos** distintos. Reutilizar a mesma collection misturando vetores de modelos diferentes deixa o índice **inconsistente** (a busca por similaridade deixa de ser confiável e podem ocorrer erros de dimensão na coluna de vetores).

## Scripts principais

| Caminho | Função |
|---------|--------|
| `src/ingest.py` | Indexa o PDF no vector store |
| `src/search.py` | Busca e montagem do prompt RAG (usado pelo chat) |
| `src/chat.py` | Loop interativo |

## Problemas comuns

- **Porta 5432 em uso** — libere a porta ou mude o mapeamento no Compose e o `DATABASE_URL`.
- **Erro de dimensão de vetor** — use outro `PG_VECTOR_COLLECTION_NAME` adequado ao modelo atual e rode o ingest de novo (ver seção acima).
