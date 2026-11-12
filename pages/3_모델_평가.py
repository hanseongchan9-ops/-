import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import (
    split_data,
    train_models,
    evaluate_model
)

st.set_page_config(
    page_title="모델 평가",
    layout="wide"
)

st.title("3. 모델 성능 평가")

st.write(
    "선형회귀 모델의 예측 성능을 평가하고 "
    "기준 모델과 비교합니다."
)

try:
    (
        df,
        X_train, y_train,
        X_valid, y_valid,
        X_test, y_test
    ) = split_data()

    model, baseline = train_models()

except Exception as e:
    st.error(f"모델 평가 오류: {e}")
    st.stop()


# -----------------------------
# 테스트 데이터 예측
# -----------------------------
test_pred = np.ravel(model.predict(X_test))
baseline_pred = np.ravel(baseline.predict(X_test))
y_test_arr = np.ravel(y_test)

model_result = evaluate_model(
    y_test_arr,
    test_pred
)

baseline_result = evaluate_model(
    y_test_arr,
    baseline_pred
)

# -----------------------------
# 데이터 분리 정보
# -----------------------------
st.header("데이터 분리")

col1, col2, col3 = st.columns(3)

col1.metric("학습 데이터", f"{len(X_train):,}개")
col2.metric("검증 데이터", f"{len(X_valid):,}개")
col3.metric("평가 데이터", f"{len(X_test):,}개")

st.caption(
    "시간 순서를 유지하여 학습 70%, "
    "검증 15%, 평가 15%로 분리했습니다."
)

# -----------------------------
# 선형회귀 성능
# -----------------------------
st.divider()
st.header("선형회귀 모델 성능")

col1, col2, col3 = st.columns(3)

col1.metric(
    "MAE",
    f"{model_result['MAE']:.2f} Wh"
)

col2.metric(
    "RMSE",
    f"{model_result['RMSE']:.2f} Wh"
)

col3.metric(
    "R²",
    f"{model_result['R2']:.4f}"
)

# -----------------------------
# 모델 비교
# -----------------------------
st.divider()
st.header("기준 모델과 성능 비교")

comparison = pd.DataFrame({
    "평가 지표": ["MAE", "RMSE", "R²"],
    "기준 모델": [
        baseline_result["MAE"],
        baseline_result["RMSE"],
        baseline_result["R2"]
    ],
    "선형회귀": [
        model_result["MAE"],
        model_result["RMSE"],
        model_result["R2"]
    ]
})

st.dataframe(
    comparison,
    use_container_width=True
)

st.caption(
    "MAE와 RMSE는 낮을수록, R²는 높을수록 "
    "일반적으로 좋은 성능을 의미합니다."
)

# -----------------------------
# 실제값과 예측값 비교
# -----------------------------
st.divider()
st.header("실제값과 예측값 비교")

result_df = pd.DataFrame({
    "실제 에너지 소비량": y_test_arr[:200],
    "예측 에너지 소비량": test_pred[:200]
})

st.line_chart(result_df)

# -----------------------------
# 오차 분석
# -----------------------------
st.divider()
st.header("예측 오차 분석")

errors = y_test_arr - test_pred

fig, ax = plt.subplots(figsize=(9, 4))

ax.hist(
    errors,
    bins=30,
    edgecolor="black"
)

ax.set_xlabel("Prediction Error (Wh)")
ax.set_ylabel("Frequency")
ax.set_title("Prediction Error Distribution")

st.pyplot(fig)
plt.close(fig)

# -----------------------------
# 해석
# -----------------------------
st.divider()
st.header("결과 해석")

if model_result["MAE"] < baseline_result["MAE"]:
    st.success(
        "선형회귀 모델의 MAE가 기준 모델보다 낮습니다. "
        "평균값만 사용하는 방식보다 예측 오차가 작았습니다."
    )
else:
    st.warning(
        "선형회귀 모델의 MAE가 기준 모델보다 낮지 않습니다. "
        "입력 변수와 모델의 적합성을 추가로 검토해야 합니다."
    )

if model_result["R2"] < 0:
    st.warning(
        "R²가 0보다 작습니다. "
        "평균값을 예측하는 기준보다도 성능이 낮을 수 있습니다."
    )
