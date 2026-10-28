from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict
from dbn import Student_DBN
import os
import json

from solver import solver  
app = FastAPI(title="Integration Dynamic Bayesian Network and SQL")

origins = [
    "http://localhost:3000",  
    "http://localhost:5500",  
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"]
)

app.mount("/static", StaticFiles(directory="static"), name="static")

JSON_FILE = "questoes.json"

def carregar_questoes_do_arquivo():
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, "r", encoding="utf-8") as f:
            content = f.read()
            if "questions =" in content:
                content = content.split("=", 1)[1].strip()
            return json.loads(content)
    return {}

@app.get("/questoes")
def listar_questoes():
    dados = carregar_questoes_do_arquivo()
    lista_plana = []
    for topico, questoes in dados.items():
        for q in questoes:
            lista_plana.append(q)
    return lista_plana

class QuizSubmission(BaseModel):
    answers: Dict[str, str]           
    feedbacks: Dict[str, str] = {}    

class NodeDataResponse(BaseModel):
    name: str
    probability: float

    class Config:
        from_attributes = True

dados = carregar_questoes_do_arquivo()
topicos = ["Lógica Proposicional"] + list(dados.keys())

student_model = Student_DBN("Lógica Proposicional", topicos, dados)
tutor_model = Student_DBN("Lógica Proposicional", topicos, dados)

@app.post("/inferencia")
def executar_inferencia(data: QuizSubmission):
    respostas_processadas = {topico: [] for topico in dados.keys()}
    feedback_recebido = {topico: [] for topico in dados.keys()}
    
    for topico, questoes in dados.items():
        for q in questoes:
            q_id = str(q["id"])
            if q_id in data.answers:
                resposta_aluno = data.answers[q_id].strip()
                gabarito_correto = q["gabarito"].strip()
                
                acertou = solver(resposta_aluno, gabarito_correto)
                
                feedback_raw = data.feedbacks.get(q_id, "50")
                try:
                    confianca_val = float(feedback_raw)
                    feedback = confianca_val/100
                except ValueError:
                    feedback = 0.01

                respostas_processadas[topico].append([acertou, 1-acertou])
                print(respostas_processadas)
                feedback_recebido[topico].append([feedback, 1-feedback])
        
    try:
        subtopicos = list(dados.keys())
        for topico in subtopicos:
            if respostas_processadas[topico]:
                if len(respostas_processadas[topico]) == 0:
                 continue
                else:
                    print(respostas_processadas[topico])
                    tutor_model.do_inference(topico, respostas_processadas[topico])
            if feedback_recebido[topico]:
                if len(feedback_recebido[topico]) == 0:
                 continue
                else:
                    print('aqui')
                    student_model.do_inference(topico, feedback_recebido[topico])
                
        print(tutor_model.get_nodes_percentage())
        return {
            "probabilidades_tutor": tutor_model.get_nodes_percentage(),
            "probabilidades_aluno": student_model.get_nodes_percentage(),
            "probabilidade_geral": tutor_model.get_nodes_percentage()["Lógica Proposicional"]
        }

    except Exception as e:
        return {
            "status": "sucesso_parcial",
            "detalhes_respostas": respostas_processadas,
            "mensagem_erro_rede": str(e),
            "probabilidades": tutor_model.get_nodes_percentage()
        }

@app.get("/")
async def read_root():
    from fastapi.responses import FileResponse
    return FileResponse("static/index.html")