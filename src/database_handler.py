import json
import os

class DatabaseHandler:
    def __init__(self, json_path: str):
        """Carrega o JSON contendo a base de dados"""
        self.json_path = json_path
        self.data = self.load_json()

    def load_json(self):
        """Carrega os dados do JSON para a memória"""
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"Arquivo JSON não encontrado: {self.json_path}")

        with open(self.json_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def search(self, query: str):
        """
        Procura uma pergunta na base de dados percorrendo todas as seções (Artigos, Títulos, Súmulas, Acórdãos).
        Retorna a resposta mais relevante encontrada.
        """
        results = []

        for section in self.data:  # Percorre as seções (Artigos, Títulos, Súmulas, Acórdãos)
            for category_name, category_content in section.items():
                for item_list in category_content:
                    for item in item_list:
                        question = item.get("Pergunta", "").strip().lower()
                        answer = item.get("Resposta", "").strip()
                        context = item.get("Contexto", "").strip()

                        # Verifica se a query está dentro da pergunta armazenada
                        if query.lower() in question:
                            results.append({
                                "categoria": category_name,
                                "pergunta": question,
                                "resposta": answer,
                                "contexto": context
                            })

        # Retorna a melhor resposta ou None se não encontrar
        return results[0] if results else None
