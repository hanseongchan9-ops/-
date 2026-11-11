
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(
    page_title="Smart Energy Predictor",
    page_icon="⚡",
    layout="wide"
)

st.title("Smart Energy Predictor")
st.caption("실내외 환경 데이터를 활용한 가정용 전자기기 에너지 소비량 예측")

st.divider()

# -----------------------------
# 데이터 불러오기
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("energydata_complete.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("energydata_complete.csv 파일을 GitHub 저장소에 업로드해 주세요.")
    st.stop()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# -----------------------------
# 데이터 전처리
# -----------------------------
df["hour"] = df["date"].dt.hour
df["dayofweek"] = df["date"].dt.dayofweek

df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

# 10분 후 에너지 소비량 생성
df["target_next"] = df["Appliances"].shift(-1)

df = df.dropna(subset=["target_next"]).copy()

features = [
    "T1", "RH_1",
    "T2", "RH_2",
    "T_out", "RH_out",
    "Windspeed", "lights",
    "hour_sin", "hour_cos",
    "dayofweek"
]

X = df[features]
y = df["target_next"]

# -----------------------------
# 시간 순서 유지 데이터 분리
# -----------------------------
n = len(df)

train_end = int(n * 0.70)
valid_end = int(n * 0.85)

X_train = X.iloc[:train_end]
y_train = y.iloc[:train_end]

X_valid = X.iloc[train_end:valid_end]
y_valid = y.iloc[train_end:valid_end]

X_test = X.iloc[valid_end:]
y_test = y.iloc[valid_end:]

# -----------------------------
# 모델 학습
# -----------------------------
@st.cache_resource
def train_models(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)

    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)

    return model, baseline

model, baseline = train_models(X_train, y_train)

# -----------------------------
# 평가 함수
# -----------------------------
def evaluate(y_true, predictions):
    mae = mean_absolute_error(y_true, predictions)
    rmse = np.sqrt(mean_squared_error(y_true, predictions))
    r2 = r2_score(y_true, predictions)

    return mae, rmse, r2

valid_pred = model.predict(X_valid)
test_pred = model.predict(X_test)
baseline_pred = baseline.predict(X_test)

# -----------------------------
# 메뉴
# -----------------------------
tab1, tab2, tab3 = st.tabs([
    "데이터 분석",
    "에너지 예측",
    "모델 평가"
])

# -----------------------------
# TAB 1: 데이터 분석
# -----------------------------
with tab1:
    st.header("데이터 분석")

    col1, col2, col3 = st.columns(3)

    col1.metric("전체 데이터", f"{len(df):,}개")
    col2.metric("입력 변수", f"{len(features)}개")
    col3.metric("측정 간격", "10분")

    st.subheader("데이터 미리보기")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("에너지 소비량 분포")

    fig, ax = plt.subplots()
    ax.hist(df["Appliances"], bins=30)
    ax.set_xlabel("Energy Consumption (Wh)")
    ax.set_ylabel("Frequency")
    ax.set_title("Appliances Energy Distribution")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("시간대별 평균 에너지 소비량")

    hourly = df.groupby("hour")["Appliances"].mean()

    fig, ax = plt.subplots()
    ax.plot(hourly.index, hourly.values, marker="o")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Energy Consumption (Wh)")
    ax.set_xticks(range(24))
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)

# -----------------------------
# TAB 2: 에너지 예측
# -----------------------------
with tab2:
    st.header("10분 후 에너지 소비량 예측")
    st.write("현재 환경 정보를 입력하면 다음 10분의 에너지 소비량을 예측합니다.")

    col1, col2 = st.columns(2)

    with col1:
        t1 = st.number_input("주방 온도 (°C)", -10.0, 40.0, 22.0)
        rh1 = st.number_input("주방 습도 (%)", 0.0, 100.0, 45.0)

        t2 = st.number_input("거실 온도 (°C)", -10.0, 40.0, 21.0)
        rh2 = st.number_input("거실 습도 (%)", 0.0, 100.0, 45.0)

        tout = st.number_input("외부 온도 (°C)", -30.0, 50.0, 12.0)
        rhout = st.number_input("외부 습도 (%)", 0.0, 100.0, 60.0)

    with col2:
        windspeed = st.number_input("풍속 (m/s)", 0.0, 30.0, 3.0)
        lights = st.number_input("조명 에너지 사용량 (Wh)", 0.0, 1000.0, 10.0)

        hour = st.slider("현재 시간", 0, 23, 12)
        dayofweek = st.selectbox(
            "요일",
            options=list(range(7)),
            format_func=lambda x: [
                "월요일", "화요일", "수요일",
                "목요일", "금요일", "토요일", "일요일"
            ][x]
        )

    if st.button("에너지 소비량 예측", type="primary"):
        input_data = pd.DataFrame([{
            "T1": t1,
            "RH_1": rh1,
            "T2": t2,
            "RH_2": rh2,
            "T_out": tout,
            "RH_out": rhout,
            "Windspeed": windspeed,
            "lights": lights,
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "dayofweek": dayofweek
        }], columns=features)

        prediction = model.predict(input_data)[0]

        st.divider()
        st.subheader("예측 결과")

        st.metric(
            label="10분 후 예상 에너지 소비량",
            value=f"{max(0, prediction):.2f} Wh"
        )

        if prediction < 0:
            st.warning(
                "모델이 음수 값을 예측했습니다. "
                "이는 선형회귀 모델의 한계일 수 있습니다."
            )

# -----------------------------
# TAB 3: 모델 평가
# -----------------------------
with tab3:
    st.header("모델 성능 평가")

    mae, rmse, r2 = evaluate(y_test, test_pred)
    b_mae, b_rmse, b_r2 = evaluate(y_test, baseline_pred)

    st.subheader("선형회귀 모델")

    col1, col2, col3 = st.columns(3)

    col1.metric("MAE", f"{mae:.2f} Wh")
    col2.metric("RMSE", f"{rmse:.2f} Wh")
    col3.metric("R²", f"{r2:.4f}")

    st.subheader("기준 모델과 성능 비교")

    comparison = pd.DataFrame({
        "모델": ["평균값 기준 모델", "선형회귀"],
        "MAE": [b_mae, mae],
        "RMSE": [b_rmse, rmse],
        "R²": [b_r2, r2]
    })

    st.dataframe(comparison, use_container_width=True)

    st.subheader("실제값과 예측값 비교")

    result = pd.DataFrame({
        "실제값": y_test.values[:200],
        "예측값": test_pred[:200]
    })

    st.line_chart(result)

    st.subheader("검증 데이터 성능")

    v_mae, v_rmse, v_r2 = evaluate(y_valid, valid_pred)

    st.write(f"검증 MAE: {v_mae:.2f} Wh")
    st.write(f"검증 RMSE: {v_rmse:.2f} Wh")
    st.write(f"검증 R²: {v_r2:.4f}")

st.divider()
st.caption("Data source: UCI Machine Learning Repository - Appliances Energy Prediction")
