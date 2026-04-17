import os

from dotenv import load_dotenv

from search import search_prompt

load_dotenv()


def _get_chat_llm():
    """Google se GOOGLE_API_KEY, senão OpenAI."""
    google_key = os.getenv("GOOGLE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    if google_key:
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = os.getenv("GOOGLE_CHAT_MODEL", "gemini-2.5-flash-lite")
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=google_key,
        )
    if openai_key:
        from langchain_openai import ChatOpenAI

        model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=model, api_key=openai_key)
    raise ValueError(
        "Defina GOOGLE_API_KEY ou OPENAI_API_KEY no .env para o chat."
    )


def _response_text(response) -> str:
    if hasattr(response, "content"):
        c = response.content
        return c if isinstance(c, str) else str(c)
    return str(response)


def main():
    try:
        llm = _get_chat_llm()
    except ValueError as e:
        print(e)
        return

    print("Chat RAG (digite 'sair' para encerrar).\n")
    while True:
        question = input("Pergunta: ").strip()
        if not question:
            print("Pergunta vazia. Digite algo ou 'sair' para encerrar.")
            continue
        if question.lower() in ("sair", "exit", "quit"):
            print("Até mais.")
            break
        try:
            prompt_text = search_prompt(question)
            response = llm.invoke(prompt_text)
            print(f"Resposta: {_response_text(response)}\n")
        except Exception as e:
            print(f"Erro: {e}\n")


if __name__ == "__main__":
    main()
