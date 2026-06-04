import json
import time
import requests
from bs4 import BeautifulSoup

f= open("dataset_articles.json", "r", encoding="utf-8")
dataset_original= json.load(f)
f.close()

def extrair_artigo_completo(url):
    "Acede ao link do artigo e tenta extrair o texto completo do artigo"

    html = requests.get(url).text
    soup = BeautifulSoup(html, "html.parser")
    div_artigo= soup.find("div", id="artigo_full")
    texto_limpo=None

    if div_artigo:
        #eliminar as tags sup que sao tags com os numeros de referencias
        for sup in div_artigo.find_all("sup"):
            sup.decompose()


        texto_limpo =div_artigo.get_text(separator=" ").strip()#get text remove todas as tags de html
        texto_limpo = texto_limpo.replace("“", '"').replace("”", '"') #trocar tipo de aspas que eram utilizadas
        texto_limpo = texto_limpo.replace("’", "'")

        while "\n\n" in texto_limpo:
            texto_limpo = texto_limpo.replace("\n\n", "\n")

        #eliminar a bibliografia
        texto_lower = texto_limpo.lower()
        if "bibliografia" in texto_lower:
            indice_bibliografia= texto_lower.find("bibliografia")
            texto_limpo= texto_limpo[:indice_bibliografia]
        
        texto_limpo= texto_limpo.replace("\n", "")
    return texto_limpo

dataset_enriquecido=[]

for i, artigo in enumerate(dataset_original):
    titulo = artigo.get("title", "Sem Título")
    autores= artigo.get("authors", "")
    abstract = artigo.get("abstract", "")
    keywords=artigo.get("keywords", "")
    data=artigo.get("publication Date", "")
    link = artigo.get("link")
    jornal=artigo.get("journal", "")
    categoria=artigo.get("category", "")

    texto_artigo_completo=None
    if link and link.startswith("http"):
        texto_artigo_completo= extrair_artigo_completo(link)

    if texto_artigo_completo and len(texto_artigo_completo) >30:
        artigo["artigo completo"] = texto_artigo_completo

    dataset_enriquecido.append(artigo)

f_out= open("dataset_enriquecido.json", "w", encoding="utf-8")
json.dump(dataset_enriquecido, f_out, ensure_ascii=False, indent=4)