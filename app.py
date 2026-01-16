import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

# --- 설정 및 보안 ---
ST_API_KEY = st.secrets["GEMINI_API_KEY"]
MODEL_NAME = "gemini-2.0-flash"  # 최신 안정화 모델 사용

st.set_page_config(page_title="PDF 요정 챗봇", layout="centered")

# --- 카카오톡 스타일 CSS ---
st.markdown("""
<style>
    .stApp { background-color: #abc1d1; }
    .chat-message {
        padding: 10px; border-radius: 10px; margin-bottom: 10px;
        display: flex; flex-direction: column;
    }
    .chat-message.user {
        background-color: #fee500; align-self: flex-end;
        color: #3c3e3f; margin-left: 20%;
    }
    .chat-message.bot {
        background-color: #ffffff; align-self: flex-start;
        color: #3c3e3f; margin-right: 20%;
    }
    .chat-bubble { padding: 8px 12px; border-radius: 15px; font-size: 14px; }
</style>
""", unsafe_allow_html=True)

# --- PDF 처리 및 RAG 엔진 구축 ---
def setup_rag(uploaded_file):
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=ST_API_KEY)
    vector_db = FAISS.from_documents(texts, embeddings)
    
    # 프롬프트 템플릿 설정 (모르는 내용은 모른다고 답변하도록 제약)
    template = """당신은 제공된 문서를 바탕으로 친절하게 답변하는 챗봇입니다.
    문서의 내용에 없는 질문이거나 확실하지 않은 경우, "죄송합니다. 해당 내용은 문서에서 찾을 수 없습니다."라고 답변하세요.
    말투는 친절하고 귀엽게 하세요.

    Context: {context}
    Question: {question}
    Answer:"""
    
    QA_PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])
    
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=ST_API_KEY, temperature=0.1)
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(),
        chain_type_kwargs={"prompt": QA_PROMPT}
    )

# --- UI 세션 관리 ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# --- 사이드바 및 파일 업로드 ---
with st.sidebar:
    st.title("💛 PDF 채팅방")
    uploaded_file = st.file_uploader("PDF 파일을 업로드해주세요", type="pdf")
    if uploaded_file:
        with st.spinner("문서를 읽고 있습니다..."):
            st.session_state.rag_chain = setup_rag(uploaded_file)
            st.success("준비 완료!")

# --- 메인 채팅 화면 ---
st.title("💬 PDF 요정")

# 채팅 내역 표시
for message in st.session_state.messages:
    role_class = "user" if message["role"] == "user" else "bot"
    st.markdown(f"""
    <div class="chat-message {role_class}">
        <div class="chat-bubble">{message["content"]}</div>
    </div>
    """, unsafe_allow_html=True)

# 사용자 입력
if prompt := st.chat_input("궁금한 것을 물어보세요!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 답변 생성
    if st.session_state.rag_chain:
        response = st.session_state.rag_chain.invoke(prompt)
        answer = response["result"]
    else:
        answer = "먼저 왼쪽에서 PDF 파일을 업로드해 주세요! 📁"

    st.session_state.messages.append({"role": "bot", "content": answer})
    st.rerun()