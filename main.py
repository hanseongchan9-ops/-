import streamlit as st

st.set_page_config(
    page_title="Smart Energy Predictor",
    page_icon="⚡",
    layout="wide"
)

st.title("Smart Energy Predictor")
st.subheader("가정용 전자기기 에너지 소비량 예측 시스템")

st.markdown("""
### 프로젝트 소개

이 프로젝트는 실내외 환경 정보와 시간 데이터를 활용하여
가정용 전자기기의 10분 후 에너지 소비량을 예측하는 시스템입니다.

온도, 습도, 풍속, 조명 사용량 등의 데이터를 분석하고
선형회귀 모델을 이용하여 에너지 소비량을 예측합니다.
""")

st.divider()

st.header("주요 기능")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("데이터 분석")
    st.write("""
    - 데이터 구조 확인
    - 에너지 소비량 분포
    - 시간대별 소비 패턴
    """)

with col2:
    st.subheader("에너지 예측")
    st.write("""
    - 실내외 환경 입력
    - 시간 및 요일 설정
    - 10분 후 소비량 예측
    """)

with col3:
    st.subheader("모델 평가")
    st.write("""
    - 선형회귀 성능 평가
    - MAE, RMSE, R² 분석
    - 기준 모델과 비교
    """)

st.divider()

st.header("프로젝트 탐구 질문")

st.info(
    "실내외 환경과 시간 정보를 활용하면 "
    "10분 후 가정용 전자기기의 에너지 소비량을 "
    "얼마나 정확하게 예측할 수 있을까?"
)

st.caption(
    "Data Source: UCI Machine Learning Repository "
    "| Appliances Energy Prediction"
)
