import spacy
import json
import math
#como o dataset é estático criamos imedicatamente os valores do tfidf para o dataset para não estar sempre a faze-lo

nlp = spacy.load("pt_core_news_sm")

f = open("dataset_enriquecido.json", "r", encoding="utf-8")
dataset_artigos= json.load(f)
f.close()

raw_collection = []

for artigo in dataset_artigos:
    titulo = artigo.get("title", "")
    resumo = artigo.get("abstract", "")
    palavras_chave = artigo.get("keywords", "")
    artigo_completo= artigo.get("artigo completo","")
    
    texto_completo = f"{titulo} {resumo} {palavras_chave} {artigo_completo}"
    
    raw_collection.append(texto_completo)


def pre_processamento(collection):
    new_collection=[]
    for d in collection:
        doc= nlp(d)
        tokens_doc = [token.text.lower() for token in doc if not token.is_punct and not token.is_stop]
        new_collection.append(tokens_doc)

    return new_collection

def tf(d):
    N = len(d)
    if N == 0:
        return {}
    res = {}
    for term in d:
        res[term] = res.get(term, 0) + 1
    return {k: v / N for k, v in res.items()}

#idf(t,D) = log(N/df)
def calcular_idf(collection, unique_terms):
    N = len(collection)
    res = {}
    for term in unique_terms:
        counter = sum(1 for d in collection if term in d)
        res[term] = math.log(N / counter, 10) if counter > 0 else 0
    return res

#tf-idf(t,d,D) = tf(t,d) * idf(t,D)
def calcular_tf_idf(collection, unique_terms, idf_values):
    res = []
    for doc in collection:
        tf_values = tf(doc)
        doc_vector = [
            tf_values[term] * idf_values[term] if term in tf_values else 0
            for term in unique_terms
        ]
        res.append(doc_vector)
    return res

#-- Construcao do modelo---

dataset_processado= pre_processamento(raw_collection)
unique_terms_global= sorted(set([term for d in dataset_processado for term in d]))

idf_values_global= calcular_idf(dataset_processado, unique_terms_global)
docs_vectors_global = calcular_tf_idf(dataset_processado, unique_terms_global, idf_values_global)

modelo_tfidf={
    "unique_terms": unique_terms_global,
    "idf_values": idf_values_global,
    "docs_vectors": docs_vectors_global
}

f_out = open("modelo_tfidf.json", "w", encoding="utf-8")
json.dump(modelo_tfidf, f_out, indent=4, ensure_ascii=False)
f_out.close()