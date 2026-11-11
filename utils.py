
import streamlit as st
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.linear_model import LinearRegression, DummyRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# 프로젝트 기본 경로
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "energydata_complete.csv"

# 사용할 입력 변수
FEATURES = [
    "T1", "RH_1",
    "T2", "RH_2",
    "T_out", "RH_out",
    "Windspeed", "lights",
    "hour_sin", "hour_cos",
    "day_sin", "day_cos"
]


# -----------------------------
# 데이터 불러오기
# -----------------------------
@st.cache_data
def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "energydata_complete.csv 파일이 없습니다."
        )

    df = pd.read_csv(DATA_PATH)

    required = [
        "date", "Appliances",
        "T1", "RH_1",
        "T2", "RH_2",
        "T_out", "RH_out",
        "Windspeed", "lights"
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"필수 데이터 열이 없습니다: {missing}"
        )

    df["date"] = pd.to_datetime(
        df["date"], errors="coerce"
    )

    df = df.dropna(subset=required).copy()

    df = df.sort_values("date").reset_index(drop=True)

    return df


# -----------------------------
# 데이터 전처리
# -----------------------------
@st.cache_data
def preprocess_data():

    df = load_data()

    # 시간 및 요일 추출
    df["hour"] = df["date"].dt.hour
    df["dayofweek"] = df["date"].dt.dayofweek

    # 시간 순환 특성 변환
    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    df["day_sin"] = np.sin(
        2 * np.pi * df["dayofweek"] / 7
    )

    df["day_cos"] = np.cos(
        2 * np.pi * df["dayofweek"] / 7
    )

    # 다음 측정 시점의 에너지 소비량
    df["target_next"] = df["Appliances"].shift(-1)

    df = df.dropna(
        subset=["target_next"]
    ).reset_index(drop=True)

    X = df[FEATURES]
    y = df["target_next"]

    return df, X, y


# -----------------------------
# 시간 순서 유지 데이터 분리
# -----------------------------
@st.cache_data
def split_data():

    df, X, y = preprocess_data()

    n = len(df)

    train_end = int(n * 0.70)
    valid_end = int(n * 0.85)

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]

    X_valid = X.iloc[train_end:valid_end]
    y_valid = y.iloc[train_end:valid_end]

    X_test = X.iloc[valid_end:]
    y_test = y.iloc[valid_end:]

    return (
        df,
        X_train, y_train,
        X_valid, y_valid,
        X_test, y_test
    )


# -----------------------------
# 모델 학습
# -----------------------------
@st.cache_resource
def train_models():

    (
        df,
        X_train, y_train,
        X_valid, y_valid,
        X_test, y_test
    ) = split_data()

    # 선형회귀
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 기준 모델
    baseline = DummyRegressor(
        strategy="mean"
    )

    baseline.fit(X_train, y_train)

    return model, baseline


# -----------------------------
# 모델 평가
# -----------------------------
def evaluate_model(y_true, predictions):

    mae = mean_absolute_error(
        y_true, predictions
    )

    rmse = np.sqrt(
        mean_squared_error(y_true, predictions)
    )

    r2 = r2_score(
        y_true, predictions
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


# -----------------------------
# 예측 입력 데이터 생성
# -----------------------------
def make_input(
    t1, rh1,
    t2, rh2,
    tout, rhout,
    windspeed, lights,
    hour, dayofweek
):

    return pd.DataFrame([{
        "T1": t1,
        "RH_1": rh1,
        "T2": t2,
        "RH_2": rh2,
        "T_out": tout,
        "RH_out": rhout,
        "Windspeed": windspeed,
        "lights": lights,
        "hour_sin": np.sin(
            2 * np.pi * hour / 24
        ),
        "hour_cos": np.cos(
            2 * np.pi * hour / 24
        ),
        "day_sin": np.sin(
            2 * np.pi * dayofweek / 7
        ),
        "day_cos": np.cos(
            2 * np.pi * dayofweek / 7
        )
    }], columns=FEATURES)
