import streamlit as st
import pandas as pd
import traceback
import sys

# ============================================================================
# FUNÇÃO PARA CALCULAR pH BASEADO NOS NUTRIENTES
# ============================================================================

def calcular_ph(dados):
    """
    Calcula o pH do solo com base nos nutrientes e matéria orgânica
    Fórmula adaptada para solos tropicais
    """
    try:
        # Fatores que influenciam o pH
        # Quanto maior a matéria orgânica, mais ácido (se não corrigido)
        # Quanto maior os cátions (Ca, Mg, K), mais alcalino
        
        # Fatores de acidez (contribuem para pH baixo)
        fator_acidez = (
            dados.get('aluminum', 0.5) * 0.3 +  # Alumínio é ácido
            dados.get('h_al', 3.5) * 0.1         # H+Al contribui para acidez
        )
        
        # Fatores de basicidade (contribuem para pH alto)
        fator_basicidade = (
            dados.get('calcium', 3.0) * 0.2 +
            dados.get('magnesium', 1.5) * 0.15 +
            dados.get('potassium', 0.25) * 0.5
        )
        
        # Matéria orgânica (pode acidificar ou tamponar)
        om = dados.get('organic_matter', 25) / 100  # Normalizar
        
        # Cálculo do pH base (valores entre 4.5 e 7.5)
        ph_base = 5.5 + (fator_basicidade - fator_acidez) - (om * 0.5)
        
        # Limitar o pH entre 4.0 e 8.0 (valores realistas)
        ph = max(4.0, min(8.0, ph_base))
        
        return round(ph, 1)
    
    except Exception:
        # Valor padrão caso erro no cálculo
        return 6.0

# ============================================================================
# TRATAMENTO DE ERROS DE IMPORTAÇÃO
# ============================================================================

# Tentativa segura de importar joblib
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    joblib = None

# Tentativa segura de importar outras bibliotecas
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

# ============================================================================
# FUNÇÃO PARA CARREGAR MODELO COM TRATAMENTO DE ERRO
# ============================================================================

@st.cache_resource
def carregar_modelo():
    """Carrega o modelo e features com tratamento de erro"""
    
    # Verificar se a biblioteca necessária está disponível
    if not JOBLIB_AVAILABLE:
        st.warning("⚠️ Biblioteca 'joblib' não encontrada. Instale com: pip install joblib")
        return None, None
    
    try:
        # Verifica se os arquivos existem
        import os
        
        if not os.path.exists('modelo.pkl'):
            st.warning("⚠️ Arquivo 'modelo.pkl' não encontrado! Coloque o modelo treinado na pasta do app.")
            return None, None
        
        if not os.path.exists('features.pkl'):
            st.warning("⚠️ Arquivo 'features.pkl' não encontrado! Coloque o arquivo de features na pasta do app.")
            return None, None
        
        modelo = joblib.load('modelo.pkl')
        features = joblib.load('features.pkl')
        
        # Mostrar quantas features o modelo espera
        st.sidebar.info(f"📊 Modelo treinado com {len(features)} features")
        
        return modelo, features
    
    except Exception as e:
        st.error(f"❌ Erro ao carregar o modelo de IA: {str(e)}")
        st.code(traceback.format_exc())
        return None, None

# Carregar modelo e features
modelo, features = carregar_modelo()

# ============================================================================
# CONFIGURACAO DA PAGINA
# ============================================================================

st.set_page_config(
    page_title="Classificador de Fertilidade do Solo - SiBCS",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CSS PERSONALIZADO - DESIGN PREMIUM MODERNO
# ============================================================================

st.markdown("""
<style>

/* ==========================================================================
   IMPORTAÇÃO DE FONTE PREMIUM
========================================================================== */

@import url('https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,300;14..32,400;14..32,600;14..32,700;14..32,800&display=swap');

/* ==========================================================================
   FUNDO PRINCIPAL - GRADIENTE SOFISTICADO
========================================================================== */

.stApp {
    background: radial-gradient(circle at 0% 0%, #0a0f1e, #060912) !important;
    font-family: 'Inter', sans-serif !important;
}

.main > div {
    background: transparent !important;
}

/* ==========================================================================
   TEXTOS GERAIS
========================================================================== */

html, body, p, span, div, li, label {
    color: #e2e8f0 !important;
}

/* ==========================================================================
   TITULOS E SUBTITULOS
========================================================================== */

h1, h2, h3, h4, h5, h6 {
    background: linear-gradient(135deg, #c084fc, #60a5fa);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em !important;
}

/* ==========================================================================
   SIDEBAR - VIDRO FOSCO (GLASSMORPHISM)
========================================================================== */

section[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.85) !important;
    backdrop-filter: blur(12px) !important;
    border-right: 1px solid rgba(96, 165, 250, 0.2) !important;
    box-shadow: 8px 0 32px rgba(0, 0, 0, 0.3) !important;
}

section[data-testid="stSidebar"] * {
    color: #f1f5f9 !important;
}

/* ==========================================================================
   INPUTS - DESIGN LUXO
========================================================================== */

.stTextInput input,
.stNumberInput input,
textarea,
input {
    background: rgba(30, 41, 59, 0.9) !important;
    color: #ffffff !important;
    border: 1px solid rgba(96, 165, 250, 0.3) !important;
    border-radius: 16px !important;
    padding: 12px 16px !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
}

.stTextInput input:focus,
.stNumberInput input:focus,
textarea:focus,
input:focus {
    border-color: #60a5fa !important;
    box-shadow: 0 0 0 3px rgba(96, 165, 250, 0.2) !important;
    outline: none !important;
}

/* ==========================================================================
   SELECTBOX PREMIUM
========================================================================== */

.stSelectbox div[data-baseweb="select"] {
    background: rgba(30, 41, 59, 0.9) !important;
    border-radius: 16px !important;
    border: 1px solid rgba(96, 165, 250, 0.3) !important;
}

.stSelectbox span {
    color: #ffffff !important;
    font-weight: 500 !important;
}

/* ==========================================================================
   BOTÕES - EFEITO NEON
========================================================================== */

.stButton button {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 40px !important;
    padding: 0.75rem 2rem !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 4px 14px 0 rgba(59, 130, 246, 0.3) !important;
}

.stButton button:hover {
    transform: translateY(-2px) scale(1.02);
    box-shadow: 0 8px 25px 0 rgba(139, 92, 246, 0.4) !important;
}

/* ==========================================================================
   CARDS PRINCIPAIS (RESULTADOS)
========================================================================== */

.result-card {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.7), rgba(30, 41, 59, 0.5)) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(96, 165, 250, 0.25) !important;
    border-radius: 28px !important;
    padding: 28px !important;
    margin-top: 20px !important;
    box-shadow: 0 20px 35px -12px rgba(0, 0, 0, 0.5) !important;
}

.result-card h2 {
    font-size: 1.8rem !important;
    margin-bottom: 12px !important;
}

/* ==========================================================================
   CAIXAS DE MÉTRICAS
========================================================================== */

.result-box {
    background: linear-gradient(145deg, rgba(15, 23, 42, 0.8), rgba(30, 41, 59, 0.6)) !important;
    backdrop-filter: blur(8px) !important;
    border: 1px solid rgba(96, 165, 250, 0.2) !important;
    border-radius: 24px !important;
    padding: 20px !important;
    margin-bottom: 18px !important;
    transition: all 0.3s ease !important;
    text-align: center !important;
}

.result-box:hover {
    transform: translateY(-5px);
    border-color: rgba(139, 92, 246, 0.5) !important;
    box-shadow: 0 12px 28px -8px rgba(139, 92, 246, 0.3) !important;
}

.result-title {
    color: #94a3b8 !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    text-transform: uppercase !important;
    margin-bottom: 12px !important;
}

.result-value {
    background: linear-gradient(135deg, #f1f5f9, #cbd5e1);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent !important;
    font-size: 2.5rem !important;
    font-weight: 900 !important;
}

/* ==========================================================================
   METRICS STREAMLIT PERSONALIZADAS
========================================================================== */

div[data-testid="stMetric"] {
    background: rgba(15, 23, 42, 0.7) !important;
    backdrop-filter: blur(8px) !important;
    border: 1px solid rgba(96, 165, 250, 0.2) !important;
    border-radius: 20px !important;
    padding: 1.2rem !important;
}

div[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}

div[data-testid="stMetricValue"] {
    background: linear-gradient(135deg, #ffffff, #94a3b8);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent !important;
    font-size: 2rem !important;
    font-weight: 900 !important;
}

/* ==========================================================================
   DATAFRAME - DESIGN MODERNO
========================================================================== */

.dataframe {
    background: rgba(15, 23, 42, 0.7) !important;
    border-radius: 20px !important;
    overflow: hidden !important;
    border: 1px solid rgba(96, 165, 250, 0.2) !important;
}

.dataframe th {
    background: linear-gradient(135deg, #1e293b, #0f172a) !important;
    color: #c084fc !important;
    font-weight: 700 !important;
    padding: 12px !important;
}

.dataframe td {
    background: rgba(30, 41, 59, 0.5) !important;
    color: #e2e8f0 !important;
    padding: 10px !important;
}

/* ==========================================================================
   ALERTAS E INFOS
========================================================================== */

.stAlert {
    border-radius: 20px !important;
    border: none !important;
    background: rgba(15, 23, 42, 0.8) !important;
    backdrop-filter: blur(8px) !important;
}

/* ==========================================================================
   EXPANDER PREMIUM
========================================================================== */

.streamlit-expanderHeader {
    background: linear-gradient(135deg, #1e293b, #0f172a) !important;
    color: #c084fc !important;
    border-radius: 16px !important;
    font-weight: 700 !important;
    border: 1px solid rgba(96, 165, 250, 0.2) !important;
}

.streamlit-expanderContent {
    background: rgba(15, 23, 42, 0.6) !important;
    border-radius: 0 0 16px 16px !important;
    border: 1px solid rgba(96, 165, 250, 0.1) !important;
    border-top: none !important;
}

/* ==========================================================================
   TABS
========================================================================== */

button[data-baseweb="tab"] {
    background: transparent !important;
    color: #94a3b8 !important;
    border-radius: 40px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
    color: white !important;
    box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3) !important;
}

/* ==========================================================================
   LINHA DIVISÓRIA
========================================================================== */

hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, #3b82f6, #8b5cf6, #c084fc, transparent) !important;
    margin: 2rem 0 !important;
}

/* ==========================================================================
   SCROLLBAR PERSONALIZADA
========================================================================== */

::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: #0f172a;
    border-radius: 10px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #60a5fa, #a78bfa);
}

/* ==========================================================================
   CARD HEADER PRINCIPAL
========================================================================== */

.main-header {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(139, 92, 246, 0.1));
    backdrop-filter: blur(12px);
    border: 1px solid rgba(96, 165, 250, 0.2);
    border-radius: 32px;
    padding: 2rem;
    margin-bottom: 2rem;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# ============================================================================
# CABECALHO PREMIUM
# ============================================================================

st.markdown("""
<div class="main-header">
    <h1 style="background: linear-gradient(135deg, #c084fc, #60a5fa); -webkit-background-clip: text; background-clip: text; color: transparent; font-size: 2.5rem; margin-bottom: 0.5rem;">🌾 Classificador Inteligente de Fertilidade do Solo</h1>
    <p style="color: #94a3b8; font-size: 1.1rem;">Baseado no Sistema Brasileiro de Classificação de Solos (SiBCS) - Embrapa</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("<div style='text-align: center; margin-bottom: 1rem;'>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/2934/2934128.png", width=70)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### 📊 Sobre o Sistema")
    st.markdown("""
    Este classificador utiliza parâmetros do **SiBCS (Embrapa)** para:
    - ✅ Avaliação da fertilidade do solo
    - ✅ Classificação agrícola
    - ✅ Recomendação de calagem
    - ✅ Relatórios técnicos
    - ✅ Cálculo de CTC e saturação por bases
    """)

    st.markdown("---")
    st.markdown("### 🤖 Status da IA")
    
    if not JOBLIB_AVAILABLE:
        st.error("⚠️ joblib não instalado")
        st.caption("Execute: pip install joblib")
    elif modelo is not None and features is not None:
        st.success("✅ IA carregada com sucesso!")
        st.caption(f"Features esperadas: {len(features)}")
        with st.expander("📋 Features do modelo"):
            for i, f in enumerate(features[:10]):
                st.caption(f"{i+1}. {f}")
            if len(features) > 10:
                st.caption(f"... e mais {len(features)-10} features")
    else:
        st.warning("⚠️ IA não disponível")
        if JOBLIB_AVAILABLE:
            st.caption("Verifique os arquivos modelo.pkl e features.pkl")
        else:
            st.caption("Instale joblib para ativar a IA")

    st.markdown("---")
    st.caption("Versão 3.0 — Design Premium")
    st.caption("Desenvolvido em Streamlit")

# ============================================================================
# SESSION STATE
# ============================================================================

if "dados_basicos" not in st.session_state:
    st.session_state.dados_basicos = {}

if "dados_calculados" not in st.session_state:
    st.session_state.dados_calculados = {}

if "dados_salvos" not in st.session_state:
    st.session_state.dados_salvos = False

# MENU
menu = st.radio(
    "Navegação",
    [
        "📊 1. Dados do Solo",
        "🌱 2. Classificação",
        "📈 3. Relatório",
        "ℹ️ 4. Métodos"
    ],
    horizontal=True,
    label_visibility="collapsed"
)

# ============================================================================
# DICIONARIO DAS CULTURAS
# ============================================================================

necessidades_culturas = {
    "Soja": {"v_desejado": 60, "ph_min": 5.5, "ph_max": 6.5, "p_min": 15, "k_min": 0.35},
    "Milho": {"v_desejado": 65, "ph_min": 5.5, "ph_max": 6.5, "p_min": 20, "k_min": 0.40},
    "Feijão": {"v_desejado": 65, "ph_min": 5.5, "ph_max": 6.5, "p_min": 20, "k_min": 0.35},
    "Café": {"v_desejado": 70, "ph_min": 5.5, "ph_max": 6.5, "p_min": 25, "k_min": 0.40},
    "Pastagem": {"v_desejado": 50, "ph_min": 5.0, "ph_max": 6.0, "p_min": 10, "k_min": 0.25},
    "Algodão": {"v_desejado": 70, "ph_min": 5.5, "ph_max": 6.5, "p_min": 25, "k_min": 0.45},
    "Cana-de-açúcar": {"v_desejado": 60, "ph_min": 5.5, "ph_max": 6.5, "p_min": 15, "k_min": 0.30},
    "Sorgo": {"v_desejado": 55, "ph_min": 5.2, "ph_max": 6.2, "p_min": 12, "k_min": 0.30},
    "Trigo": {"v_desejado": 65, "ph_min": 5.5, "ph_max": 6.5, "p_min": 18, "k_min": 0.35},
    "Tomate": {"v_desejado": 80, "ph_min": 6.0, "ph_max": 6.8, "p_min": 30, "k_min": 0.50},
    "Eucalipto": {"v_desejado": 45, "ph_min": 5.0, "ph_max": 6.0, "p_min": 8, "k_min": 0.20},
    "Citrus": {"v_desejado": 70, "ph_min": 5.5, "ph_max": 6.5, "p_min": 20, "k_min": 0.35},
    "Arroz": {"v_desejado": 50, "ph_min": 5.0, "ph_max": 6.0, "p_min": 10, "k_min": 0.25}
}

# ============================================================================
# FUNÇÃO PARA PREDIÇÃO DA IA
# ============================================================================

def fazer_predicao_ia(dados):
    """Faz a predição usando o modelo de IA carregado com todas as features"""
    
    if not JOBLIB_AVAILABLE:
        return None, "Biblioteca joblib não instalada"
    
    if modelo is None or features is None:
        return None, "Modelo não disponível"
    
    try:
        features_disponiveis = {
            "nitrogen": dados.get("nitrogen", 0),
            "phosphorus": dados.get("phosphorus", 0),
            "potassium": dados.get("potassium", 0),
            "ph": dados.get("ph", 6.0),
            "organic_matter": dados.get("organic_matter", 25),
            "bulk_density": dados.get("bulk_density", 1.2),
            "sand": dados.get("sand", 350),
            "silt": dados.get("silt", 300),
            "clay": dados.get("clay", 350),
            "calcium": dados.get("calcium", 3.0),
            "magnesium": dados.get("magnesium", 1.5),
            "aluminum": dados.get("aluminum", 0.5),
            "h_al": dados.get("h_al", 3.5),
            "particle_density": dados.get("particle_density", 2.65)
        }
        
        valores = []
        for feature in features:
            if feature in features_disponiveis:
                valores.append(features_disponiveis[feature])
            else:
                valores.append(0)
        
        entrada_ia = pd.DataFrame([valores], columns=features)
        predicao = modelo.predict(entrada_ia)

        def mapear_classe_fertilidade(valor):
            mapeamento = {
                0: "🔴 BAIXA FERTILIDADE",
                1: "🟢 ALTA FERTILIDADE"
            }
            if isinstance(valor, str):
                return valor
            return mapeamento.get(valor, f"Classe {valor}")
        
        if hasattr(modelo, 'classes_'):
            classe_original = modelo.classes_[predicao[0]]
            classe_legivel = mapear_classe_fertilidade(classe_original)
            return classe_legivel, "Sucesso"
        else:
            classe_legivel = mapear_classe_fertilidade(predicao[0])
            return classe_legivel, "Sucesso"
    
    except Exception as e:
        return None, f"Erro na predição: {str(e)}"

# ============================================================================
# ABA 1 - DADOS DO SOLO
# ============================================================================

if menu == "📊 1. Dados do Solo":
    st.markdown("## 📋 Dados Básicos do Solo")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🧪 Macronutrientes")
        nitrogen = st.text_input("🌱 Nitrogênio (N) - mg/dm³", value="30.0", key="n_input")
        phosphorus = st.text_input("🔴 Fósforo (P) - mg/dm³", value="20.0", key="p_input")
        potassium = st.text_input("🟡 Potássio (K+) - cmolc/dm³", value="0.25", key="k_input")
        st.markdown("### 🌿 Matéria Orgânica")
        organic_matter = st.text_input("🌱 Matéria Orgânica (g/kg)", value="25.0", key="om_input")

    with col2:
        st.markdown("### ⚖️ Densidade")
        bulk_density = st.text_input("📦 Densidade do Solo (g/cm³)", value="1.20", key="bd_input")
        particle_density = st.text_input("💎 Densidade de Partícula (g/cm³)", value="2.65", key="pd_input")
        st.markdown("### 🏺 Textura")
        sand = st.text_input("🏖️ Areia (g/kg)", value="350", key="sand_input")
        silt = st.text_input("🏞️ Silte (g/kg)", value="300", key="silt_input")
        clay = st.text_input("🏺 Argila (g/kg)", value="350", key="clay_input")

        try:
            soma_textura = float(sand.replace(",", ".")) + float(silt.replace(",", ".")) + float(clay.replace(",", "."))
            if abs(soma_textura - 1000) > 10:
                st.warning(f"⚠️ Soma da textura = {soma_textura:.0f} g/kg. O ideal é 1000 g/kg.")
        except:
            pass

    st.markdown("---")

    if st.button("✅ SALVAR DADOS BÁSICOS", key="salvar_basicos"):
        try:
            st.session_state.dados_basicos = {
                "nitrogen": float(nitrogen.replace(",", ".")),
                "phosphorus": float(phosphorus.replace(",", ".")),
                "potassium": float(potassium.replace(",", ".")),
                "organic_matter": float(organic_matter.replace(",", ".")),
                "bulk_density": float(bulk_density.replace(",", ".")),
                "particle_density": float(particle_density.replace(",", ".")),
                "sand": float(sand.replace(",", ".")),
                "silt": float(silt.replace(",", ".")),
                "clay": float(clay.replace(",", "."))
            }
            st.session_state.dados_salvos = True
            st.success("✅ Dados básicos salvos com sucesso! Agora vá para a aba 'Classificação'.")
        except ValueError as e:
            st.error(f"❌ Verifique os valores numéricos inseridos. Erro: {str(e)}")
        except Exception as e:
            st.error(f"❌ Erro inesperado: {str(e)}")

# ============================================================================
# ABA 2 - CLASSIFICAÇÃO
# ============================================================================

elif menu == "🌱 2. Classificação":
    if "dados_basicos" not in st.session_state or not st.session_state.dados_basicos:
        st.warning("⚠️ Preencha e salve os dados na ABA 1.")
        st.stop()

    st.markdown("## 🌱 Classificação da Fertilidade")

    col1, col2 = st.columns(2)

    with col1:
        aluminum = st.text_input("⚠️ Alumínio (Al³⁺) - cmolc/dm³", value="0.50")
        h_al = st.text_input("📊 H + Al - cmolc/dm³", value="3.50")

    with col2:
        calcium = st.text_input("🥛 Cálcio (Ca²⁺)", value="3.00")
        magnesium = st.text_input("🧂 Magnésio (Mg²⁺)", value="1.50")
        cultura = st.selectbox("🌾 Cultura", list(necessidades_culturas.keys()))

    st.markdown("---")

    try:
        dados = st.session_state.dados_basicos.copy()
        dados["aluminum"] = float(aluminum.replace(",", "."))
        dados["h_al"] = float(h_al.replace(",", "."))
        dados["calcium"] = float(calcium.replace(",", "."))
        dados["magnesium"] = float(magnesium.replace(",", "."))

        dados["ph"] = calcular_ph(dados)
        st.info(f"🧪 pH calculado automaticamente: **{dados['ph']}**")

        sb = dados["calcium"] + dados["magnesium"] + dados["potassium"]
        ctc_efetiva = sb + dados["aluminum"]
        ctc_potencial = sb + dados["h_al"]

        v_percent = (sb / ctc_potencial) * 100 if ctc_potencial > 0 else 0
        m_percent = (dados["aluminum"] / ctc_efetiva) * 100 if ctc_efetiva > 0 else 0

        st.session_state.dados_calculados = dados
        st.session_state.sb = sb
        st.session_state.ctc_potencial = ctc_potencial
        st.session_state.v_percent = v_percent
        st.session_state.m_percent = m_percent
        st.session_state.cultura = cultura

        st.success("✅ Classificação realizada com sucesso!")

        st.markdown("---")
        st.markdown("## 🤖 Inteligência Artificial")
        
        if JOBLIB_AVAILABLE:
            if modelo is not None and features is not None:
                with st.spinner("🔄 IA processando os dados do solo..."):
                    predicao, status = fazer_predicao_ia(dados)
                
                if predicao is not None:
                    st.success(f"🌾 **Classe prevista pela IA:** {predicao}")
                    
                    with st.expander("ℹ️ Sobre a classificação da IA"):
                        st.markdown(f"""
                        **Modelo:** RandomForestClassifier  
                        **Features utilizadas:** {len(features)} parâmetros do solo  
                        
                        A IA analisou os seguintes parâmetros principais:
                        - Nitrogênio (N): {dados['nitrogen']} mg/dm³
                        - Fósforo (P): {dados['phosphorus']} mg/dm³
                        - Potássio (K+): {dados['potassium']} cmolc/dm³
                        - pH: {dados['ph']}
                        - Matéria Orgânica: {dados['organic_matter']} g/kg
                        - Densidade: {dados['bulk_density']} g/cm³
                        - Textura: Areia {dados['sand']} | Silte {dados['silt']} | Argila {dados['clay']} g/kg
                        """)
                else:
                    st.warning(f"⚠️ IA não pôde fazer a predição: {status}")
            else:
                st.warning("⚠️ Modelo de IA não disponível. Verifique os arquivos 'modelo.pkl' e 'features.pkl'.")
        else:
            st.info("ℹ️ Funcionalidade de IA não disponível. Instale a biblioteca 'joblib' para ativar a classificação por IA.")

    except ValueError as e:
        st.error(f"❌ Verifique os valores digitados. Erro: {str(e)}")
    except Exception as e:
        st.error(f"❌ Erro inesperado: {str(e)}")
        st.code(traceback.format_exc())

# ============================================================================
# ABA 3 - RELATÓRIO
# ============================================================================

elif menu == "📈 3. Relatório":
    st.markdown("## 📈 Relatório Técnico")

    if "v_percent" not in st.session_state or "dados_calculados" not in st.session_state or "cultura" not in st.session_state:
        st.warning("⚠️ Execute primeiro a classificação na ABA 2.")
        st.stop()

    dados = st.session_state.dados_calculados
    sb = st.session_state.sb
    ctc_potencial = st.session_state.ctc_potencial
    v_percent = st.session_state.v_percent
    m_percent = st.session_state.m_percent
    cultura = st.session_state.cultura

    relatorio = pd.DataFrame({
        "Parâmetro": [
            "pH", "Nitrogênio (N)", "Fósforo (P)", "Potássio (K+)",
            "Cálcio (Ca²⁺)", "Magnésio (Mg²⁺)", "Alumínio (Al³⁺)", "H + Al",
            "Soma de Bases (SB)", "CTC Potencial", "Saturação por Bases (V%)",
            "Saturação por Al (m%)", "Matéria Orgânica", "Densidade do Solo"
        ],
        "Valor": [
            f"{dados['ph']:.1f}", f"{dados['nitrogen']:.1f} mg/dm³",
            f"{dados['phosphorus']:.1f} mg/dm³", f"{dados['potassium']:.2f} cmolc/dm³",
            f"{dados['calcium']:.2f} cmolc/dm³", f"{dados['magnesium']:.2f} cmolc/dm³",
            f"{dados['aluminum']:.2f} cmolc/dm³", f"{dados['h_al']:.2f} cmolc/dm³",
            f"{sb:.2f} cmolc/dm³", f"{ctc_potencial:.2f} cmolc/dm³",
            f"{v_percent:.1f}%", f"{m_percent:.1f}%",
            f"{dados['organic_matter']:.1f} g/kg", f"{dados['bulk_density']:.2f} g/cm³"
        ]
    })

    st.dataframe(relatorio, hide_index=True, use_container_width=True)

    csv = relatorio.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Baixar Relatório (CSV)",
        data=csv,
        file_name="relatorio_solo.csv",
        mime="text/csv",
        key="download_csv"
    )

    st.markdown("---")
    st.markdown("## 🌾 Recomendações Agronômicas")
    st.success(f"✅ Cultura selecionada: {cultura}")

    v2 = necessidades_culturas[cultura]["v_desejado"]
    nc = max(((v2 - v_percent) * ctc_potencial) / 100, 0)
    prnt = 80
    nc_corrigida = nc * (100 / prnt)
    gesso = nc_corrigida * 0.5 if dados["clay"] >= 350 else 0

    st.info(f"🪨 Aplicar {nc_corrigida:.2f} t/ha de calcário com PRNT {prnt}%")
    
    if gesso > 0:
        st.warning(f"🌱 Recomenda-se gessagem de {gesso:.2f} t/ha")
    else:
        st.success("✅ Gessagem não necessária")

    if dados["phosphorus"] < 15:
        st.error("🔴 Necessária adubação fosfatada")
    else:
        st.success("✅ Fósforo em nível adequado")

    if dados["potassium"] < 0.30:
        st.error("🔴 Necessária adubação potássica")
    else:
        st.success("✅ Potássio em nível adequado")

# ============================================================================
# ABA 4 - MÉTODOS
# ============================================================================

elif menu == "ℹ️ 4. Métodos":
    st.markdown("## ℹ️ Métodos Utilizados")

    with st.expander("📊 Saturação por Bases (V%)"):
        st.markdown("### Fórmula:")
        st.latex(r"V\% = \frac{SB}{CTC} \times 100")
        st.markdown("Onde: SB = Soma de Bases, CTC = Capacidade de Troca de Cátions")

    with st.expander("🔬 Saturação por Alumínio (m%)"):
        st.markdown("### Fórmula:")
        st.latex(r"m\% = \frac{Al^{3+}}{CTC\ efetiva} \times 100")
        st.markdown("Onde: Al³⁺ = Alumínio trocável, CTC efetiva = SB + Al³⁺")

    with st.expander("🌾 Interpretação Agronômica"):
        st.markdown("""
        | V% | Interpretação |
        |---|---|---|
        | > 70 | Muito fértil |
        | 50-70 | Fértil |
        | 25-50 | Distrófico |
        | < 25 | Álico |
        """)

    with st.expander("🪨 Cálculo da Calagem"):
        st.markdown("### Fórmula utilizada:")
        st.latex(r"NC = \frac{(V_2 - V_1) \times CTC}{100}")
        st.markdown("Onde: NC = Necessidade de calcário, V₂ = Saturação desejada, V₁ = Saturação atual, CTC = Capacidade de troca catiônica")

    with st.expander("🌱 Cálculo da Gessagem"):
        st.markdown("### Critério utilizado:")
        st.markdown("- Solos com argila > 350 g/kg")
        st.markdown("- Gessagem = 50% da dose de calcário")

# ============================================================================
# RODAPÉ
# ============================================================================

st.markdown("---")
st.caption("© 2026 - Classificador de Fertilidade do Solo | Créditos ao SiBCS - Embrapa | Ferramenta acadêmica | Direitos reservados")
