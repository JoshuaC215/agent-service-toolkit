import os
import openai
from dotenv import load_dotenv

load_dotenv()


class ExternalAgent:
    def __init__(self):
        """Inicializa o agente de busca externa com a API do OpenAI."""
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Chave da API do OpenAI não encontrada.")

    def get_response(self, question: str):
        """Consulta a API do OpenAI para obter uma resposta."""

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",  # Usa o modelo configurado
                messages=[{"role": "user", "content": question}],
                api_key=self.api_key
            )

            generated_text = response["choices"][0]["message"]["content"]

            return {
                "categoria": "Resposta Externa (GPT)",
                "pergunta": question,
                "resposta": generated_text,
                "contexto": "Resposta gerada pela API OpenAI.",
            }

        except Exception as e:
            return {
                "categoria": "Erro",
                "pergunta": question,
                "resposta": f"Erro ao acessar a API: {str(e)}",
                "contexto": "",
            }


# Exemplo de uso
if __name__ == "__main__":
    agent = ExternalAgent()
    question = "Quais são meus direitos ao comprar um produto defeituoso?"
    response = agent.get_response(question)

    print("\n Resposta da API:")
    print(f" **Categoria**: {response['categoria']}")
    print(f" **Pergunta**: {response['pergunta']}")
    print(f" **Resposta**: {response['resposta']}")
    print(f" **Contexto**: {response['contexto']}")
