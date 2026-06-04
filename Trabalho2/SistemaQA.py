from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from RI_TFIDF import obter_artigo_mais_relevante_tfidf
from RI_SBERT import obter_artigo_mais_relevante_sbert
import json

f=open("dataset_enriquecido.json", "r", encoding="utf-8")
dataset_artigos= json.load(f)
f.close()

model_name = "pierreguillou/bert-base-cased-squad-v1.1-portuguese"

#tokenizer transforma o texto em tokens
tokenizer = AutoTokenizer.from_pretrained(model_name) #from pretainned faz download automatico do tokenizer associado ao modelo
#carrega a rede neuronal treinada
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)


def executar_ri_qa(query_pesquisa, pergunta_qa, metodo="tfidf"): #usa o modelo tfidf por defeito caso utilizador nao tenha escolhido nenhum
    
    if metodo.lower()=="sbert":
        lista_ranking= obter_artigo_mais_relevante_sbert(query_pesquisa)
    else:
        lista_ranking= obter_artigo_mais_relevante_tfidf(query_pesquisa)
    
    if not lista_ranking:
        return{
            "sucesso": False,
            "erro": "Nenhum artigo relevante foi encontrado para esta pesquisa.",
            "titulo_artigo":None,
            "resposta_bert":None,
            "score_confianca":0.0,
            "ranking_completo": []
        }
    
    # Tenta extrair resposta dos top 3 artigos, fica com a de maior confiança
    top_artigos = lista_ranking[:3]
    melhor_resultado = None

    for artigo, score_ri in top_artigos:
        contexto = artigo.get("artigo completo", "").strip()
        if not contexto:
            contexto = artigo.get("abstract", "").strip()
        if not contexto:
            continue  # salta artigos sem texto

        try:
            resultado = qa_pipeline(question=pergunta_qa, context=contexto)
            if melhor_resultado is None or resultado["score"] > melhor_resultado["score"]:
                melhor_resultado = {
                    "score": resultado["score"],
                    "answer": resultado["answer"],
                    "titulo": artigo.get("title", "Título Desconhecido")
                }
        except Exception:
            continue  # se um artigo falhar, tenta o próximo

    if not melhor_resultado:
        return {
            "sucesso": False,
            "erro": "Nenhum dos artigos relevantes continha texto suficiente para extrair uma resposta.",
            "titulo_artigo": None,
            "resposta_bert": None,
            "score_confianca": 0.0,
            "ranking_completo": lista_ranking
        }

    return {
        "sucesso": True,
        "erro": None,
        "titulo_artigo": melhor_resultado["titulo"],
        "resposta_bert": melhor_resultado["answer"],
        "score_confianca": round(melhor_resultado["score"] * 100, 2),
        "ranking_completo": lista_ranking
    }