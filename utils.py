
import streamlit as st
import pandas as pd
import numpy as np

from pathlib import Path

# -----------------------------
# 기본 설정
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "energydata_complete.csv"

FEATURES = [
    "T1", "RH_1",
    "T2", "RH_2",
    "T_out", "RH_out",
    "Windspeed", "lights",
    "hour_sin", "hour_cos",
    "day_sin", "day_cos"
]


# -----------------------------
# 선형회귀 모델 직접 구현
# -----------------------------
class LinearRegressionModel:

    def __init__(self):
        self.coefficients = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        # 절편을 위한 열 추가
        X_with_intercept = np.column_stack([
            np.ones(len(X)),
            X
        ])

        # 최소제곱법으로 계수 계산
        self.coefficients = np.linalg.lstsq(
            X_with_intercept,
            y,
            rcond=None
        )[0]

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)

        X_with_intercept = np.column_stack([
            np.ones(len(X)),
            X
        ])

        return X_with_intercept @ self.coefficients


# -----------------------------
# 기준 모델
# -----------------------------
class MeanBaselineModel:

    def fit(self, X, y):
        self.mean_value = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(
            len(X),
            self.mean_value,
            dtype=float
        )


# -----------------------------
# 데이터 불러오기
# -----------------------------
@st.cache_data
def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "energydata_complete.csv 파일이 최상위 폴더에 없습니다."
        )

    df = pd.read_csv(DATA_PATH)

    required = [
        "date", "Appliances",
        "T1", "RH_1",
        "T2", "RH_2",
        "T_out", "RH_out",
        "Windspeed", "lights"
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"CSV에 필요한 열이 없습니다: {missing}"
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    # 숫자형 데이터 변환
    numeric_columns = [
        col for col in required
        if col != "date"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna(
        subset=required
    )

    df = df.sort_values(
        "date"
    ).reset_index(drop=True)

    if len(df) < 100:
        raise ValueError(
            "학습에 사용할 수 있는 데이터가 부족합니다."
        )

    return df


# -----------------------------
# 데이터 전처리
# -----------------------------
@st.cache_data
def preprocess_data():

    df = load_data().copy()

    # 시간과 요일 추출
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
    df["target_next"] = (
        df["Appliances"].shift(-1)
    )

    df = df.dropna(
        subset=FEATURES + ["target_next"]
    ).reset_index(drop=True)

    X = df[FEATURES].astype(float)
    y = df["target_next"].astype(float)

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

    model = LinearRegressionModel()
    model.fit(X_train, y_train)

    baseline = MeanBaselineModel()
    baseline.fit(X_train, y_train)

    return model, baseline


# -----------------------------
# 모델 평가
# -----------------------------
def evaluate_model(y_true, predictions):

    actual = np.asarray(
        y_true,
        dtype=float
    )

    predicted = np.asarray(
        predictions,
        dtype=float
    )

    errors = actual - predicted

    mae = np.mean(
        np.abs(errors)
    )

    rmse = np.sqrt(
        np.mean(errors ** 2)
    )

    ss_res = np.sum(
        errors ** 2
    )

    ss_tot = np.sum(
        (actual - np.mean(actual)) ** 2
    )

    if ss_tot == 0:
        r2 = 0.0
    else:
        r2 = 1 - (ss_res / ss_tot)

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2)
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
