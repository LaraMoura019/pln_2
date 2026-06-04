import json
import torch
import math
from sentence_transformers import SentenceTransformer, util

f_dataset= open("dataset_enriquecido.json", "r", encoding="utf-8")
dataset_artigos= json.load(f_dataset)
f_dataset.close()

f_modelo= open("modelo_sbert.json", "r", encoding="utf-8")
modelo_sbert= json.load(f_modelo)
f_modelo.close()

#transformamos em tensores PyTorch para ser mais rápido
docs_vectors_global = torch.tensor(modelo_sbert["doc_vectors"])

#para interpretar a query
modelo_sbert_q=SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def ranking(query):
    vetor_query= modelo_sbert_q.encode(query, convert_to_tensor=True)

    scores= util.cos_sim(vetor_query, docs_vectors_global)[0]

    scores_mapeados= []
    for i, score in enumerate(scores):
        scores_mapeados.append((dataset_artigos[i], score.item())) #.item() para transformar de tensor para float
    
    return sorted(scores_mapeados, key=lambda x: x[1], reverse=True)

def obter_artigo_mais_relevante_sbert(query):
    lista_ranking= ranking(query)
    if lista_ranking:
        return lista_ranking
    return []