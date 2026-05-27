import os
import streamlit as st
from openai import AzureOpenAI
from dotenv import load_dotenv

# 로컬 테스트용 .env 로드 (스트림릿 클라우드 배포 시에는 웹의 Secrets 설정 사용)
load_dotenv()

# 1. 페이지 및 레이아웃 설정
st.set_page_config(page_title="과일 전문가 AI 챗봇", page_icon="🍓", layout="centered")
st.title("🍓 과일 전문가 AI 챗봇")
st.caption("과일에 대해 물어보시면 맛과 영양성분을 귀엽게 알려드려요! 😊")

# 2. 사이드바 제어 파라미터 (기존 함수의 인자들을 UI로 연동)
with st.sidebar:
    st.header("⚙️ 챗봇 매개변수 설정")
    message_cnt = st.slider("기억할 대화 턴 수 (Turn)", min_value=1, max_value=10, value=3)
    max_tokens = st.slider("Max Tokens", min_value=100, max_value=2000, value=800)
    temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1)
    top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.95, step=0.05)
    
    # 대화 초기화 버튼
    if st.button("대화 내역 초기화 🧹", use_container_width=True):
        st.session_state.chat_prompt = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "너는 과일전문가야. 사용자가 과일에 대해서 질문하면 맛과 영양성분 등에 대해서 귀여운 어투로 대답해줘. 답변의 길이는 200자 이내로"}]
            }
        ]
        st.rerun()

# 3. Azure OpenAI 클라이언트 초기화 (캐싱을 통해 매번 재생성되는 것 방지)
@st.cache_resource
def init_azure_client():
    endpoint = os.getenv("AZURE_OAI_ENDPOINT")
    subscription_key = os.getenv("AZURE_OAI_KEY")
    
    if not endpoint or not subscription_key:
        st.error("⚠️ 환경 변수(AZURE_OAI_ENDPOINT, AZURE_OAI_KEY)가 설정되지 않았습니다.")
        st.stop()
        
    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=subscription_key,
        api_version="2025-01-01-preview",
    )

client = init_azure_client()
deployment = os.getenv("AZURE_OAI_DEPLOYMENT", "gpt-4o") # 기본값 지정

# 4. 세션 상태를 이용한 대화 내역 저장소 초기화
if "chat_prompt" not in st.session_state:
    st.session_state.chat_prompt = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "너는 과일전문가야. 사용자가 과일에 대해서 질문하면 맛과 영양성분 등에 대해서 귀여운 어투로 대답해줘. 답변의 길이는 200자 이내로"}]
        }
    ]

# 5. 기존 대화 기록 화면에 출력 (System 메시지는 숨김)
for msg in st.session_state.chat_prompt:
    if msg["role"] == "system":
        continue
    
    with st.chat_message(msg["role"]):
        for content_item in msg["content"]:
            if content_item["type"] == "text":
                st.write(content_item["text"])

# 6. 사용자 입력 창 처리 (st.chat_input 사용)
if user_input := st.chat_input("과일에 대해 무엇이든 물어보세요!"):
    
    # 대화 내역 개수 제어 로직 (기존 콘솔 코드와 동일)
    message_limit = message_cnt * 2
    if len(st.session_state.chat_prompt) > (1 + message_limit):
        st.session_state.chat_prompt = [st.session_state.chat_prompt[0]] + st.session_state.chat_prompt[-message_limit:]
        
    # 사용자가 입력한 메시지 화면에 즉시 표시 및 세션에 추가
    with st.chat_message("user"):
        st.write(user_input)
        
    st.session_state.chat_prompt.append({
        "role": "user",
        "content": [{"type": "text", "text": user_input}]
    })
    
    # 7. AI 답변 생성 및 스트리밍 효과(선택)를 위한 로딩 스피너 표시
    with st.chat_message("assistant"):
        with st.spinner("과일 박사가 맛있는 답변을 준비 중이에요... 🍉"):
            try:
                completion = client.chat.completions.create(
                    model=deployment,
                    messages=st.session_state.chat_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stream=False # 스트리밍 활성화 시 True로 변경 가능
                )
                
                ai_response = completion.choices[0].message.content
                st.write(ai_response)
                
                # AI 답변 대화 내역에 저장
                st.session_state.chat_prompt.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": ai_response}]
                })
                
            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")
