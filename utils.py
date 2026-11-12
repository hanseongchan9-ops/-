
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

        if X.ndim != 2 or y.ndim != 1:
            raise ValueError("입력 데이터의 형태가 올바르지 않습니다.")

        if len(X) != len(y) or len(X) == 0:
            raise ValueError("학습 데이터의 개수가 올바르지 않습니다.")

        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("학습 데이터에 유효하지 않은 값이 있습니다.")

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
        if self.coefficients is None:
            raise ValueError("모델이 아직 학습되지 않았습니다.")

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("예측 입력 데이터는 2차원이어야 합니다.")

        if X.shape[1] != len(self.coefficients) - 1:
            raise ValueError("학습 데이터와 예측 데이터의 변수 개수가 다릅니다.")

        if not np.isfinite(X).all():
            raise ValueError("예측 데이터에 유효하지 않은 값이 있습니다.")

        X_with_intercept = np.column_stack([
            np.ones(len(X)),
            X
        ])

        return X_with_intercept @ self.coefficients


# -----------------------------
# 기준 모델
# -----------------------------
class MeanBaselineModel:

    def __init__(self):
        self.mean_value = None

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)

        if y.ndim != 1 or len(y) == 0:
            raise ValueError("기준 모델의 학습 데이터가 올바르지 않습니다.")

        if not np.isfinite(y).all():
            raise ValueError("기준 모델의 학습 데이터에 유효하지 않은 값이 있습니다.")

        self.mean_value = float(np.mean(y))

        return self

    def predict(self, X):
        if self.mean_value is None:
            raise ValueError("기준 모델이 아직 학습되지 않았습니다.")

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

    # 날짜 데이터 변환
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

    # 무한대 및 결측치 처리
    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna(
        subset=required
    )

    # 시간 순서 정렬
    df = df.sort_values(
        "date"
    ).reset_index(drop=True)

    # 중복된 측정 시각 확인
    if df["date"].duplicated().any():
        raise ValueError(
            "CSV에 중복된 측정 시각이 있습니다. "
            "데이터를 확인해야 합니다."
        )

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

    # 정확히 10분 뒤의 에너지 소비량을 정답으로 설정
    target_df = df[
        ["date", "Appliances"]
    ].copy()

    target_df = target_df.rename(
        columns={
            "date": "target_time",
            "Appliances": "target_next"
        }
    )

    # 현재 시각에서 10분 뒤의 데이터와 연결
    df["target_time"] = (
        df["date"] + pd.Timedelta(minutes=10)
    )

    df = df.merge(
        target_df,
        on="target_time",
        how="inner",
        validate="one_to_one"
    )

    # 불필요한 시간 열 제거
    df = df.drop(
        columns=["target_time"]
    )

    # 유효한 데이터만 사용
    df = df.dropna(
        subset=FEATURES + ["target_next"]
    ).reset_index(drop=True)

    if len(df) < 100:
        raise ValueError(
            "10분 간격으로 연결할 수 있는 데이터가 부족합니다."
        )

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

    if min(
        len(X_train),
        len(X_valid),
        len(X_test)
    ) == 0:
        raise ValueError(
            "학습, 검증, 평가 데이터 분리에 실패했습니다."
        )

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

    # 선형회귀 모델
    model = LinearRegressionModel()
    model.fit(X_train, y_train)

    # 기준 모델
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

    if actual.ndim != 1 or predicted.ndim != 1:
        raise ValueError("평가 데이터는 1차원이어야 합니다.")

    if len(actual) != len(predicted) or len(actual) == 0:
        raise ValueError("실제값과 예측값의 개수가 일치하지 않습니다.")

    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("평가 데이터에 유효하지 않은 값이 있습니다.")

    errors = actual - predicted

    # 평균 절대 오차
    mae = np.mean(
        np.abs(errors)
    )

    # 평균 제곱근 오차
    rmse = np.sqrt(
        np.mean(errors ** 2)
    )

    # 결정계수
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

    values = [
        t1, rh1,
        t2, rh2,
        tout, rhout,
        windspeed, lights
    ]

    if not np.isfinite(values).all():
        raise ValueError("입력값에 유효하지 않은 값이 있습니다.")

    if not 0 <= hour <= 23:
        raise ValueError("시간은 0~23 사이여야 합니다.")

    if not 0 <= dayofweek <= 6:
        raise ValueError("요일은 0~6 사이여야 합니다.")

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
