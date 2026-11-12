import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from utils import load_data, preprocess_data

st.set_page_config(
    page_title="데이터 분석",
    layout="wide"
)

st.title("1. 데이터 분석")
st.write("가정용 전자기기 에너지 소비 데이터의 구조와 특징을 분석합니다.")

try:
    raw_df = load_data()
    raw_df["date"] = pd.to_datetime(raw_df["date"])
    df, X, y = preprocess_data()

except Exception as e:
    st.error(f"데이터 분석 오류: {e}")
    st.stop()


# -----------------------------
# 기본 데이터 정보
# -----------------------------
st.header("데이터 기본 정보")

col1, col2, col3 = st.columns(3)

col1.metric("전체 데이터", f"{len(raw_df):,}개")
col2.metric("입력 변수", f"{len(X.columns)}개")
col3.metric("정답 변수", "1개")

st.caption(
    f"측정 기간: {raw_df['date'].min()} ~ {raw_df['date'].max()}"
)

# -----------------------------
# 데이터 미리보기
# -----------------------------
st.divider()
st.header("데이터 미리보기")

st.dataframe(
    raw_df.head(10),
    use_container_width=True
)

# -----------------------------
# 변수 정보
# -----------------------------
st.divider()
st.header("변수 정보")

variable_info = {
    "T1": "주방 온도 (°C)",
    "RH_1": "주방 습도 (%)",
    "T2": "거실 온도 (°C)",
    "RH_2": "거실 습도 (%)",
    "T_out": "외부 온도 (°C)",
    "RH_out": "외부 습도 (%)",
    "Windspeed": "풍속 (m/s)",
    "lights": "조명 에너지 사용량 (Wh)",
    "Appliances": "전자기기 에너지 사용량 (Wh)"
}

info_df = raw_df[
    list(variable_info.keys())
].describe().T

info_df["설명"] = info_df.index.map(variable_info)

st.dataframe(
    info_df,
    use_container_width=True
)

# -----------------------------
# 결측치 확인
# -----------------------------
st.divider()
st.header("결측치 확인")

missing = raw_df.isnull().sum()

st.dataframe(
    missing.rename("결측치 개수"),
    use_container_width=True
)

# -----------------------------
# 에너지 소비량 분포
# -----------------------------
st.divider()
st.header("에너지 소비량 분포")

fig, ax = plt.subplots(figsize=(9, 4))

ax.hist(
    raw_df["Appliances"],
    bins=30,
    edgecolor="black"
)

ax.set_xlabel("Energy Consumption (Wh)")
ax.set_ylabel("Frequency")
ax.set_title("Appliances Energy Distribution")

st.pyplot(fig)
plt.close(fig)

# -----------------------------
# 시간대별 소비량
# -----------------------------
st.divider()
st.header("시간대별 평균 에너지 소비량")

hourly = raw_df.groupby(
    raw_df["date"].dt.hour
)["Appliances"].mean()

fig, ax = plt.subplots(figsize=(9, 4))

ax.plot(
    hourly.index,
    hourly.values,
    marker="o"
)

ax.set_xlabel("Hour")
ax.set_ylabel("Energy Consumption (Wh)")
ax.set_xticks(range(24))
ax.grid(True)

st.pyplot(fig)
plt.close(fig)

st.info(
    "시간대별 평균 소비량을 비교하면 "
    "가정용 전자기기의 사용 패턴을 파악할 수 있습니다."
)
