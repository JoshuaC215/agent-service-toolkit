import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.database_handler import DatabaseHandler
from src.agents.similarity_agent import SimilarityAgent


class InternalAgent:
    def __init__(self, database_path="data/database.json"):
        """Inicializa o agente interno com a base de dados"""
        self.db = DatabaseHandler(database_path)
        self.similarity_agent = SimilarityAgent()

    def process_query(self, question: str):
        """Obtém uma resposta do banco de dados interno e envia para o agente de similaridade"""
        results = self.db.search(question)

        if results:
            formatted_responses = []
            for res in results:
                formatted_responses.append({
                    "categoria": res["categoria"],
                    "pergunta": res["pergunta"],
                    "resposta": res["resposta"],
                    "contexto": res["contexto"]
                })
            return formatted_responses
        return [{"categoria": "Interno", "pergunta": question, "resposta": "Nenhuma resposta encontrada na base interna.", "contexto": ""}]


# Exemplo de uso
if __name__ == "__main__":
    agent = InternalAgent()
    question = "Comprei um eletrônico com garantia de um ano, mas ele apresentou defeito após seis meses."
    response = agent.process_query(question)
    print(response)
