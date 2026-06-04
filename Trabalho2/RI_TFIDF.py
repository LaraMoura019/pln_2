import spacy
import json
import math

nlp = spacy.load("pt_core_news_sm")

f_artigos = open("dataset_enriquecido.json", "r", encoding="utf-8")
dataset_artigos= json.load(f_artigos)
f_artigos.close()

f_modelo = open("modelo_tfidf.json", "r", encoding="utf-8")
modelo_tfidf= json.load(f_modelo)
f_modelo.close()

unique_terms_global= modelo_tfidf["unique_terms"]
idf_values_global= modelo_tfidf["idf_values"]
docs_vectors_global= modelo_tfidf["docs_vectors"]


def tf(d):
    N=len(d)
    res= {}
    if N == 0:
        return {}

    for term in d:
        if  term in res:
            res[term] +=1
        else:
            res[term] = 1
    res ={k: v/N for k,v in res.items()}
    return res
    #output: {"termo": freq_relativa}


def preprocessamento_query(q):
    q_nlp=nlp(q)
    tokens=[token.text.lower() for token in q_nlp if not token.is_punct and not token.is_stop]  

    return tokens

def vetor_query(query):
    query_tokens= preprocessamento_query(query)
    tf_query= tf(query_tokens)

    return [
        tf_query[term] * idf_values_global[term] if term in tf_query else 0
        for term in unique_terms_global
    ]

def similaridade_coseno(query_vector, doc_vector):
    prod=0
    for term_q, term_d in zip(query_vector, doc_vector):
        prod += term_q * term_d

    norm_q=0
    for term in query_vector:
        norm_q += term ** 2
    norm_q = math.sqrt(norm_q)

    norm_d=0
    for term in doc_vector:
        norm_d += term ** 2
    norm_d = math.sqrt(norm_d)

    if norm_q == 0 or norm_d == 0:
        return 0

    res = prod / (norm_q * norm_d)
    return res

def ranking(query):
    query_vector=vetor_query(query)
    scores =[]
    
    for i, doc_vector in enumerate(docs_vectors_global):
        score=similaridade_coseno(query_vector, doc_vector)
        scores.append((dataset_artigos[i], score))
        
    ranking= sorted(scores, key= lambda x: x[1], reverse=True)
    return ranking

def obter_artigo_mais_relevante_tfidf(query_utilizador):
    return ranking(query_utilizador)  # devolve lista de (artigo_dict, score)