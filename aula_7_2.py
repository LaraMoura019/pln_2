from flask import Flask, render_template, request, redirect, url_for
import json
import re

app = Flask(__name__)

f_db = open ("./dicionario_medico.json","r", encoding="utf-8")
db =  json.load (f_db)

@app.get("/")
def home_page():
    return render_template("home.html")

@app.get("/conceitos")
def listar_conceitos():
    pagina=request.args.get('pagina',1,type=int)
    letra_filtro = request.args.get('letra', '')
    termo = request.args.get('termo_pesquisa', '')
    word_boundary = request.args.get('word_boundary', '')
    case_sensitive = request.args.get('case_sensitive', '')
    por_pagina=20

    inicio =(pagina-1)*por_pagina
    fim = por_pagina+inicio
    conceitos=list(db.keys())

    if letra_filtro:
        conceitos = [c for c in conceitos if c.upper().startswith(letra_filtro.upper())]

    conceitos_por_pagina=conceitos[inicio:fim]
    total_paginas = (len(conceitos) + por_pagina - 1) // por_pagina

    conceitos_formatados = {}
    for c in conceitos_por_pagina:
        if termo in c and case_sensitive:
            if not word_boundary:
                c_f=c.replace(termo, f'<strong>{termo}</strong>')
                conceitos_formatados[c]=c_f

            elif word_boundary:
                c_f = re.sub(rf'\b{termo}\b', f'<strong>{termo}</strong>', c)
                conceitos_formatados[c]=c_f

        elif termo.lower() in c.lower() and not case_sensitive:
            if not word_boundary:
                c_f=c.replace(termo, f'<strong>{termo}</strong>')
                conceitos_formatados[c]=c_f

            elif word_boundary:
               c_f = re.sub(rf'\b{termo}\b', f'<strong>{termo}</strong>', c, flags=re.IGNORECASE)
               conceitos_formatados[c]=c_f

        else:
            conceitos_formatados[c]=c
            
    return render_template('conceitos.html', 
                           conceitos=conceitos_formatados, 
                           pagina_atual=pagina, 
                           total_paginas=total_paginas,
                           letra_ativa=letra_filtro,
                           termo_pesquisa=termo,
                           word_boundary=word_boundary,
                           case_sensitive=case_sensitive)

@app.get("/conceitos/<designacao>") # <designacao> estamos a passar um parametro
def conceito(designacao):
    if designacao in db:
        descricao = db[designacao]
        return render_template("conceito.html", designacao=designacao, descricao=descricao)
    else:
        return render_template("error.html", erro="O conceito introduzido não existe")

#o get aparece a informacao no url e o post nao
@app.post("/conceitos")
def adicionar_conceito():
    #colocamos a variavel que pusemos na variavel name do html
    designacao=request.form["designacao"]
    descricao=request.form["descricao"]
    db[designacao]=descricao
    f_out = open("bd.json","w",encoding="utf-8")
    json.dump(db,f_out,indent=4, ensure_ascii=False)
    f_out.close()

    return redirect(url_for('listar_conceitos'))

@app.delete("/conceitos/<designacao>")
def apagar_conceito(designacao):
    del db[designacao]
    f_out = open("bd.json","w",encoding="utf-8")
    json.dump(db,f_out,indent=4, ensure_ascii=False)
    f_out.close()

    return url_for('listar_conceitos')

@app.get("/api/conceitos")
def conceitos_api():
    return db

@app.errorhandler(404)
def pagina_nao_encontrada(e):
    return render_template('error.html'), 404

@app.get("/tabela")
def visualizar_tabela():
    return render_template('tabela.html', conceitos=db)

app.run(host="localhost", port=4002, debug=True)