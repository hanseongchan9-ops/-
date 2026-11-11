
import streamlit as st

from utils import train_models, make_input

st.set_page_config(
    page_title="에너지 예측",
    layout="wide"
)

st.title("2. 에너지 소비량 예측")

st.write(
    "현재 환경 정보를 입력하면 "
    "다음 10분의 에너지 소비량을 예측합니다."
)

try:
    model, baseline = train_models()

except Exception as e:
    st.error(f"모델 학습 오류: {e}")
    st.stop()


# -----------------------------
# 입력 화면
# -----------------------------
st.header("환경 정보 입력")

col1, col2 = st.columns(2)

with col1:
    st.subheader("실내 환경")

    t1 = st.number_input(
        "주방 온도 (°C)",
        value=22.0
    )

    rh1 = st.number_input(
        "주방 습도 (%)",
        min_value=0.0,
        max_value=100.0,
        value=45.0
    )

    t2 = st.number_input(
        "거실 온도 (°C)",
        value=21.0
    )

    rh2 = st.number_input(
        "거실 습도 (%)",
        min_value=0.0,
        max_value=100.0,
        value=45.0
    )

with col2:
    st.subheader("외부 환경")

    tout = st.number_input(
        "외부 온도 (°C)",
        value=12.0
    )

    rhout = st.number_input(
        "외부 습도 (%)",
        min_value=0.0,
        max_value=100.0,
        value=60.0
    )

    windspeed = st.number_input(
        "풍속 (m/s)",
        min_value=0.0,
        value=3.0
    )

    lights = st.number_input(
        "조명 에너지 사용량 (Wh)",
        min_value=0.0,
        value=10.0
    )

# -----------------------------
# 시간 정보
# -----------------------------
st.divider()
st.subheader("시간 정보")

col3, col4 = st.columns(2)

with col3:
    hour = st.slider(
        "현재 시간",
        min_value=0,
        max_value=23,
        value=12
    )

with col4:
    day_names = [
        "월요일", "화요일", "수요일",
        "목요일", "금요일", "토요일", "일요일"
    ]

    day_name = st.selectbox(
        "요일",
        day_names
    )

    dayofweek = day_names.index(day_name)

# -----------------------------
# 예측 실행
# -----------------------------
st.divider()

if st.button(
    "10분 후 에너지 소비량 예측",
    type="primary",
    use_container_width=True
):

    input_data = make_input(
        t1, rh1,
        t2, rh2,
        tout, rhout,
        windspeed, lights,
        hour, dayofweek
    )

    prediction = float(
        model.predict(input_data)[0]
    )

    st.subheader("예측 결과")

    st.metric(
        "10분 후 예상 에너지 소비량",
        f"{prediction:.2f} Wh"
    )

    if prediction < 0:
        st.warning(
            "음수 예측값이 발생했습니다. "
            "선형회귀 모델의 예측 한계이므로 "
            "실제 소비량으로 해석할 수 없습니다."
        )

    st.caption(
        "이 결과는 학습 데이터의 관계를 바탕으로 계산한 "
        "모델 예측값이며 실제 측정값과 다를 수 있습니다."
    )
