import sys
import os
import re
from collections import Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.database_handler import DatabaseHandler
from .external_agent import ExternalAgent


def tokenize(text):
    """Tokeniza a string em palavras únicas, removendo pontuação e convertendo para minúsculas."""
    words = re.findall(r'\b\w+\b', text.lower())
    return set(words)


def tanimoto_similarity(text1, text2):
    """Calcula a similaridade de Tanimoto entre duas strings."""
    set1 = tokenize(text1)
    set2 = tokenize(text2)

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    if union == 0:
        return 0.0  # Evita divisão por zero

    return intersection / union


class SimilarityAgent:
    def __init__(self, database_path="data/database.json"):
        """Inicializa o agente de similaridade e carrega os outros agentes."""
        self.internal_agent = DatabaseHandler(database_path)
        self.external_agent = ExternalAgent()

    def get_best_response(self, question: str):
        """Obtém a melhor resposta comparando a base interna e a API externa."""
        
        # Busca na base interna
        internal_results = self.internal_agent.search(question)
        internal_response = (
            internal_results[0] if internal_results else {"resposta": "Não encontrado", "contexto": ""}
        )

        # Busca na API externa (GPT)
        external_response = self.external_agent.get_response(question)

        # Compara as respostas usando Tanimoto
        similarity_score = tanimoto_similarity(internal_response["resposta"], external_response["resposta"])

        # Decide qual resposta retornar
        if similarity_score > 0.5:  # Ajuste do threshold conforme necessário
            return internal_response
        return external_response


# Exemplo de uso
if __name__ == "__main__":
    agent = SimilarityAgent()
    question = "Comprei um eletrônico com garantia de um ano, mas ele apresentou defeito após seis meses."
    best_response = agent.get_best_response(question)

    print("\n Melhor resposta encontrada:")
    print(f" **Categoria**: {best_response.get('categoria', 'Não informado')}")
    print(f" **Pergunta**: {best_response.get('pergunta', 'Não informado')}")
    print(f" **Resposta**: {best_response['resposta']}")
    print(f" **Contexto**: {best_response['contexto']}")
