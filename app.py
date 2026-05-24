# ============================================
# SISTEMA CLÍNICO INTEGRAL COM IA
# Protótipo Acadêmico — Google Colab
# ============================================

# ============================================
# IMPORTAÇÃO DAS BIBLIOTECAS
# ============================================

import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ============================================
# BASE DE DADOS CLÍNICA SIMULADA
# ============================================

dados = {
    
    'febre': [1,1,0,0,1,0,1,0,1,0],
    
    'tosse': [1,1,0,0,1,0,1,1,0,0],
    
    'dor_toracica': [0,1,1,0,0,1,1,0,0,1],
    
    'fadiga': [1,1,1,0,1,0,1,0,1,0],
    
    'leucocitose': [1,1,0,0,1,0,1,0,1,0],
    
    'saturacao_baixa': [1,1,0,0,1,0,1,0,0,0],
    
    'diagnostico': [
        'Pneumonia',
        'Pneumonia',
        'Infarto',
        'Saudável',
        'Pneumonia',
        'Infarto',
        'COVID-19',
        'Bronquite',
        'COVID-19',
        'Saudável'
    ]
}

df = pd.DataFrame(dados)

print("BASE DE DADOS CLÍNICA")
print(df)

# ============================================
# DEFINIÇÃO DAS VARIÁVEIS
# ============================================

X = df.drop('diagnostico', axis=1)

y = df['diagnostico']

# ============================================
# DIVISÃO DE TREINO E TESTE
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ============================================
# TREINAMENTO DA IA
# ============================================

modelo = DecisionTreeClassifier()

modelo.fit(X_train, y_train)

# ============================================
# AVALIAÇÃO DO MODELO
# ============================================

predicoes = modelo.predict(X_test)

acuracia = accuracy_score(y_test, predicoes)

print("\nACURÁCIA DO MODELO:")
print(acuracia)

# ============================================
# SIMULAÇÃO DE NOVO PACIENTE
# ============================================

print("\n================================")
print("ANÁLISE CLÍNICA DO PACIENTE")
print("================================")

# EXEMPLO:
# febre
# tosse
# dor_toracica
# fadiga
# leucocitose
# saturacao_baixa

novo_paciente = [[
    1,  # febre
    1,  # tosse
    0,  # dor torácica
    1,  # fadiga
    1,  # leucocitose
    1   # saturação baixa
]]

resultado = modelo.predict(novo_paciente)

print("\nDIAGNÓSTICO SUGERIDO:")
print(resultado[0])

# ============================================
# ALERTAS CLÍNICOS INTELIGENTES
# ============================================

print("\nALERTAS CLÍNICOS:")

if novo_paciente[0][5] == 1:
    print("- Saturação baixa detectada")

if novo_paciente[0][4] == 1:
    print("- Possível processo infeccioso")

if novo_paciente[0][0] == 1 and novo_paciente[0][1] == 1:
    print("- Síndrome respiratória identificada")

# ============================================
# HISTÓRICO DO PACIENTE
# ============================================

historico = {

    "Nome": "Paciente Teste",
    
    "Idade": 45,
    
    "Sexo": "Feminino",
    
    "Sintomas": [
        "Febre",
        "Tosse",
        "Fadiga"
    ],
    
    "Diagnóstico IA": resultado[0]
}

print("\n================================")
print("PRONTUÁRIO ELETRÔNICO")
print("================================")

for chave, valor in historico.items():
    print(f"{chave}: {valor}")

# ============================================
# EXPORTAÇÃO DE RELATÓRIO
# ============================================

relatorio = pd.DataFrame([historico])

relatorio.to_csv("relatorio_clinico.csv", index=False)

print("\nRELATÓRIO CLÍNICO EXPORTADO!")

# ============================================
# STREAMLIT (INTERFACE VISUAL)
# ============================================

codigo_streamlit = '''

import streamlit as st
import pandas as pd

st.title("Sistema Clínico Integral com IA")

st.header("Inserção de Sintomas")

febre = st.selectbox("Febre", [0,1])
tosse = st.selectbox("Tosse", [0,1])
dor_toracica = st.selectbox("Dor Torácica", [0,1])
fadiga = st.selectbox("Fadiga", [0,1])
leucocitose = st.selectbox("Leucocitose", [0,1])
saturacao_baixa = st.selectbox("Saturação Baixa", [0,1])

if st.button("Analisar Paciente"):

    entrada = [[
        febre,
        tosse,
        dor_toracica,
        fadiga,
        leucocitose,
        saturacao_baixa
    ]]

    resultado = modelo.predict(entrada)

    st.success(f"Diagnóstico sugerido: {resultado[0]}")

'''

with open("app_streamlit.py", "w") as arquivo:
    arquivo.write(codigo_streamlit)

print("\nARQUIVO STREAMLIT CRIADO!")

# ============================================
# FIM DO SISTEMA
# ============================================

print("\n================================")
print("SISTEMA CLÍNICO INTEGRAL FINALIZADO")
print("================================")