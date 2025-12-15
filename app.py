import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Inicializar API
app = FastAPI(
    title="API de Qualidade de Vinho",
    description="Uma API simples para prever se um vinho é bom ou ruim baseada em ML.",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite que qualquer site acesse (para desenvolvimento local é ok)
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos os métodos (GET, POST, etc)
    allow_headers=["*"],
)

# Carregar os arquivos .pkl
try:
    modelo = joblib.load('modelo_vinho.pkl')
    scaler = joblib.load('scaler_vinho.pkl')
    print("Modelo e Scaler carregados com sucesso!")
except FileNotFoundError:
    print("ERRO: Arquivos .pkl não encontrados. Verifique se estão na mesma pasta que app.py")

# Definir input
class VinhoInput(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

# Rota de Previsão 
@app.post("/predict")
def prever_qualidade(vinho: VinhoInput):
    
    # Converter o input para um dicionário
    dados_dict = vinho.dict()
    
    # Mapear os nomes das variáveis para os nomes originais do CSV (com espaços)
    mapa_colunas = {
        "fixed_acidity": "fixed acidity",
        "volatile_acidity": "volatile acidity",
        "citric_acid": "citric acid",
        "residual_sugar": "residual sugar",
        "chlorides": "chlorides",
        "free_sulfur_dioxide": "free sulfur dioxide",
        "total_sulfur_dioxide": "total sulfur dioxide",
        "density": "density",
        "pH": "pH",
        "sulphates": "sulphates",
        "alcohol": "alcohol"
    }
    
    # Criar DataFrame com os nomes corrigidos
    dados_df = pd.DataFrame([dados_dict]).rename(columns=mapa_colunas)
    
    # Aplicar a escala (scaler) nos dados recebidos
    # O modelo aprendeu com dados transformados, então precisamos transformar o input também
    dados_escalados = scaler.transform(dados_df)
    
    # Fazer a previsão
    predicao = modelo.predict(dados_escalados)[0]
    probabilidade = modelo.predict_proba(dados_escalados)[0][1] # Chance de ser "1" (Bom)
    
    # Preparar a resposta
    resultado_texto = "Bom" if predicao == 1 else "Ruim"
    
    return {
        "resultado": resultado_texto,
        "probabilidade": f"{probabilidade:.2%}",
        "input_recebido": dados_dict
    }

# Rota de teste simples (só para ver se a API está viva)
@app.get("/")
def home():
    return {"mensagem": "API de Vinhos está rodando!"}