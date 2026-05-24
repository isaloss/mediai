# ============================================
# SISTEMA CLÍNICO INTEGRAL COM IA
# Protótipo Acadêmico — Streamlit
# ============================================

import pandas as pd
import streamlit as st

from fpdf import FPDF
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# ============================================
# BASE DE DADOS CLÍNICA SIMULADA
# ============================================

DADOS = {
    'febre':           [1, 1, 0, 0, 1, 0, 1, 0, 1, 0],
    'tosse':           [1, 1, 0, 0, 1, 0, 1, 1, 0, 0],
    'dor_toracica':    [0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
    'fadiga':          [1, 1, 1, 0, 1, 0, 1, 0, 1, 0],
    'leucocitose':     [1, 1, 0, 0, 1, 0, 1, 0, 1, 0],
    'saturacao_baixa': [1, 1, 0, 0, 1, 0, 1, 0, 0, 0],
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
        'Saudável',
    ],
}

# Ordem das features usada no treino e na predição (precisa ser a mesma).
FEATURES = ['febre', 'tosse', 'dor_toracica', 'fadiga', 'leucocitose', 'saturacao_baixa']


# ============================================
# TREINAMENTO DA IA (cacheado: roda uma vez)
# ============================================

@st.cache_resource
def treinar_modelo():
    df = pd.DataFrame(DADOS)

    X = df[FEATURES]
    y = df['diagnostico']

    modelo = DecisionTreeClassifier(random_state=42)
    modelo.fit(X, y)

    # Acurácia sobre a própria base. A base é pequena (10 linhas), então
    # isso é só uma referência didática, não uma métrica de generalização.
    acuracia = accuracy_score(y, modelo.predict(X))

    return modelo, acuracia


modelo, acuracia = treinar_modelo()


# ============================================
# GERAÇÃO DO PRONTUÁRIO EM PDF
# ============================================

def gerar_pdf(prontuario, acuracia):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 12, 'Prontuário Eletrônico', new_x='LMARGIN', new_y='NEXT')

    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 6, 'Protótipo acadêmico - dados simulados, não use para decisão clínica real.',
             new_x='LMARGIN', new_y='NEXT')
    pdf.ln(4)

    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Helvetica', '', 12)

    for campo, valor in prontuario.items():
        pdf.set_font('Helvetica', 'B', 12)
        pdf.cell(45, 9, f'{campo}:')
        pdf.set_font('Helvetica', '', 12)
        pdf.multi_cell(0, 9, str(valor), new_x='LMARGIN', new_y='NEXT')

    pdf.ln(4)
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 6, f'Acurácia do modelo sobre a base de treino: {acuracia:.0%}',
             new_x='LMARGIN', new_y='NEXT')

    # fpdf2 devolve bytearray; convertemos para bytes para o download_button.
    return bytes(pdf.output())


# ============================================
# INTERFACE — INSERÇÃO MANUAL DE SINTOMAS
# ============================================

st.title('Sistema Clínico Integral com IA')
st.caption('Protótipo acadêmico — dados simulados, não use para decisão clínica real.')

st.header('Dados do paciente')

col1, col2 = st.columns(2)

with col1:
    nome = st.text_input('Nome', 'Paciente Teste')
    idade = st.number_input('Idade', min_value=0, max_value=120, value=45)

with col2:
    sexo = st.selectbox('Sexo', ['Feminino', 'Masculino', 'Outro'])

st.header('Sintomas e exames')

# Checkbox devolve True/False; convertemos para 1/0 que é o que o modelo espera.
febre           = st.checkbox('Febre')
tosse           = st.checkbox('Tosse')
dor_toracica    = st.checkbox('Dor torácica')
fadiga          = st.checkbox('Fadiga')
leucocitose     = st.checkbox('Leucocitose')
saturacao_baixa = st.checkbox('Saturação baixa')

if st.button('Analisar paciente', type='primary'):
    valores = {
        'febre': int(febre),
        'tosse': int(tosse),
        'dor_toracica': int(dor_toracica),
        'fadiga': int(fadiga),
        'leucocitose': int(leucocitose),
        'saturacao_baixa': int(saturacao_baixa),
    }

    # DataFrame com a mesma ordem/nomes de coluna do treino evita warning do sklearn.
    entrada = pd.DataFrame([valores])[FEATURES]

    resultado = modelo.predict(entrada)[0]

    # ----- Resultado na tela -----
    st.subheader('Diagnóstico sugerido')
    st.success(resultado)
    st.caption(f'Acurácia do modelo sobre a base de treino: {acuracia:.0%}')

    # ----- Alertas clínicos -----
    alertas = []

    if saturacao_baixa:
        alertas.append('Saturação baixa detectada')

    if leucocitose:
        alertas.append('Possível processo infeccioso')

    if febre and tosse:
        alertas.append('Síndrome respiratória identificada')

    if alertas:
        st.subheader('Alertas clínicos')

        for alerta in alertas:
            st.warning(alerta)

    # ----- Prontuário eletrônico -----
    st.subheader('Prontuário eletrônico')

    sintomas_presentes = [
        nome_sintoma.replace('_', ' ').capitalize()
        for nome_sintoma, presente in valores.items()
        if presente == 1
    ]

    prontuario = {
        'Nome': nome,
        'Idade': idade,
        'Sexo': sexo,
        'Sintomas': ', '.join(sintomas_presentes) or 'Nenhum',
        'Diagnóstico IA': resultado,
    }

    st.table(pd.DataFrame([prontuario]))

    # ----- Download do relatório -----
    pdf = gerar_pdf(prontuario, acuracia)

    st.download_button(
        'Exportar relatório (PDF)',
        data=pdf,
        file_name='relatorio_clinico.pdf',
        mime='application/pdf',
    )
