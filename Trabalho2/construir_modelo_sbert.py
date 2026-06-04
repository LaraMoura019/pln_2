import json
import re
from collections import Counter
from itertools import combinations
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.losses import CosineSimilarityLoss
from sentence_transformers import SentenceTransformerTrainingArguments
from sentence_transformers.sentence_transformer.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction
from sentence_transformers import SentenceTransformerTrainer

#guarda automaticamente os embeddings de cada artigo

f = open("dataset_enriquecido.json", "r", encoding="utf-8")
dataset_artigos= json.load(f)
f.close()

model_name= "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

model= SentenceTransformer(model_name)

raw_collection = []
for artigo in dataset_artigos:
    titulo = artigo.get("title", "")
    resumo = artigo.get("abstract", "")
    palavras_chave = artigo.get("keywords", "")
    artigo_completo = artigo.get("artigo completo", "")
    
    texto_completo = f"{titulo} {resumo} {palavras_chave} {artigo_completo}"
    raw_collection.append(texto_completo)

#converte os textps em embeddings automaticamente
embeddings_docs= model.encode(raw_collection, convert_to_tensor= False)

modelo_sbert_json={"doc_vectors":[embedding.tolist() for embedding in embeddings_docs]} #tolist serve para transformar os numeros do modelo numa lista normal que o JSON aceita

f_out= open("modelo_sbert.json", "w", encoding="utf-8")
json.dump(modelo_sbert_json, f_out, ensure_ascii=False, indent=4)
f_out.close()


















filtered_data =[]
for artigo in dataset_artigos:
    if artigo["abstract"] and len(artigo["abstract"]) > 50 and artigo["keywords"] and len(artigo["keywords"])>1: 
        filtered_data.append(artigo)


def normalize_keywords(s):
    s = s.lower().strip()
    keywords = re.split(r"[,;]",s)
    return [k.strip() for k in keywords] 

abstract_pairs = []
for p1, p2 in combinations(filtered_data, 2):
    
    p2["abstract"]
    k1 = normalize_keywords(p1["keywords"])
    k2 = normalize_keywords(p2["keywords"])
    score = len(set(k1) & set(k2))
    abstract_pairs.append((p1["abstract"], p2["abstract"], score))

majority_class = [pair for pair in abstract_pairs if pair[2] == 0]
minority_class = [pair for pair in abstract_pairs if pair[2] != 0]

undersampled_majority_class = resample(majority_class,
                            replace=False,
                            n_samples= len(minority_class), 
                            random_state=42) 


balanced_pairs= undersampled_majority_class + minority_class

def normalize_scores(score):
    if score == 0:
        return 0
    if score == 1:
        return 0.5
    if score == 2:
        return 0.75
    if score >= 3:
        return 0.85
    
balanced_pairs_norm = [(a1,a2, normalize_scores(score)) for a1, a2, score in balanced_pairs]

#----Train/test splt---
scores = [score for _, _ , score in balanced_pairs_norm]

train_data, test_data = train_test_split(
    balanced_pairs_norm,
    test_size=0.2,
    random_state=42,
    stratify = scores
)

train_dataset = Dataset.from_list([{"abstract": a1, "abstract2": a2, "score": score} for a1, a2, score in train_data])
test_dataset = Dataset.from_list([{"abstract1": a1, "abstract2": a2, "score": score} for a1, a2, score in test_data])


model = SentenceTransformer("neuralmind/bert-base-portuguese-cased")
loss = CosineSimilarityLoss(model)

args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir="models/meu_modelo_sbert",
    # Optional training parameters:
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    warmup_steps=0.1,
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    # Optional tracking/debugging parameters:
    eval_strategy="epoch",
    eval_steps=100,
    save_strategy="epoch",
    save_steps=100,
    save_total_limit=2,
    logging_steps=100,
)

dev_evaluator = EmbeddingSimilarityEvaluator(
    sentences1=test_dataset["abstract1"],
    sentences2=test_dataset["abstract2"],
    scores=test_dataset["score"],
    main_similarity=SimilarityFunction.COSINE
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    loss=loss,
    evaluator=dev_evaluator,
)

trainer.train()