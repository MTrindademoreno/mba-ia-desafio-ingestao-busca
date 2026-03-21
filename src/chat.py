from search import search_prompt


def main():
    chain = search_prompt()
    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    while True:
        question = input("Pergunta: ").strip()
        if not question:
            print("Pergunta vazia. Digite algo ou 'sair' para encerrar.")
            continue
        if question.lower() in ("sair", "exit", "quit"):
            print("Até mais.")
            break
        response = chain.invoke(question)
        print(f"Resposta: {response}")


if __name__ == "__main__":
    main()

