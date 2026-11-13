"""데이터 로드 · 요약 · 누수 점검용 함수 모음."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "hotel_bookings.csv"
TARGET = "is_canceled"

# 1일차 분석에서 확인한 누수 열 (근거는 notebooks/01_eda_baseline.ipynb 참고)
LEAKY_COLS = ["reservation_status", "reservation_status_date", "assigned_room_type"]

# 2일차: 예약 이후에 채워지는 것으로 판단한 열 (근거는 notebooks/02_validation_models.ipynb 참고)
POST_BOOKING_COLS = ["required_car_parking_spaces", "booking_changes"]

# 2일차: 시간 기준 분할에서 일반화를 해치는 날짜 열 (연도는 미래에 처음 보는 값, 주차는 해마다 어긋남)
UNSTABLE_DATE_COLS = ["arrival_date_year", "arrival_date_week_number"]


def load_data(path=DATA_PATH):
    """원본 CSV를 읽는다. 'NULL' 문자열은 결측치로 처리한다."""
    return pd.read_csv(path, na_values=["NULL"])


def summarize(df):
    """열별 dtype, 결측 수/비율, 고유값 수, 예시 값을 표로 반환한다."""
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "n_missing": df.isna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(2),
        "n_unique": df.nunique(),
        "example": df.iloc[0],
    })


def target_ratio(df, target=TARGET):
    """타깃 클래스별 건수와 비율."""
    counts = df[target].value_counts().sort_index()
    return pd.DataFrame({"count": counts, "ratio": (counts / counts.sum()).round(4)})


def single_feature_auc(df, target=TARGET, random_state=42):
    """열 하나만으로 타깃을 얼마나 잘 맞히는지 AUC로 측정한다.

    데이터를 반으로 나눠 앞쪽에서 범주별 타깃 평균을 학습하고 뒤쪽에서 평가하므로,
    고유값이 많은 열이 외워서 점수가 부풀려지는 것을 막는다.
    수치형 열은 값 자체를 점수로 쓰고, 방향은 max(auc, 1-auc)로 무시한다.
    AUC가 1에 가까운 열은 정답을 그대로 담고 있을 가능성이 높다.
    """
    fit, ev = train_test_split(df, test_size=0.5, stratify=df[target], random_state=random_state)
    prior = fit[target].mean()
    rows = []
    for col in df.columns.drop(target):
        if pd.api.types.is_numeric_dtype(df[col]):
            score = ev[col].fillna(ev[col].median())
        else:
            means = fit.groupby(fit[col].fillna("__NA__"))[target].mean()
            score = ev[col].fillna("__NA__").map(means).fillna(prior)
        if score.nunique() < 2:
            auc = 0.5
        else:
            auc = roc_auc_score(ev[target], score)
        rows.append({"column": col, "auc": max(auc, 1 - auc), "n_unique": df[col].nunique()})
    return pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)


def arrival_date(df):
    """연/월/일 열을 합쳐 도착일(datetime)을 만든다."""
    return pd.to_datetime(
        df["arrival_date_year"].astype(str) + "-"
        + df["arrival_date_month"] + "-"
        + df["arrival_date_day_of_month"].astype(str),
        format="%Y-%B-%d",
    )


def duplicate_group_id(df, cols=None):
    """모든 값이 똑같은 행끼리 같은 번호를 매긴다 (그룹 단위 분할용).

    행마다 다른 번호를 붙이면 겉으로만 달라질 뿐 피처는 그대로라 누수를 못 막는다.
    똑같은 행에 같은 번호를 줘야 GroupKFold 등으로 한쪽 세트에만 몰아넣을 수 있다.
    """
    cols = list(df.columns) if cols is None else cols
    return df.groupby(cols, dropna=False, sort=False).ngroup()


def time_split(df, cutoffs):
    """도착일 기준으로 구간을 나눠 인덱스 목록을 반환한다.

    cutoffs=["2017-01-01", "2017-05-01"] → [~2016-12, 2017-01~04, 2017-05~] 세 구간.
    같은 묶음(똑같은 행)은 도착일이 같으므로 자연히 한 구간에만 들어간다.
    """
    d = arrival_date(df)
    edges = [pd.Timestamp.min] + [pd.Timestamp(c) for c in cutoffs] + [pd.Timestamp.max]
    return [df.index[(d >= lo) & (d < hi)] for lo, hi in zip(edges[:-1], edges[1:])]


def add_features(df):
    """예약 시점에 알 수 있는 값만으로 파생 피처를 만든다."""
    df = df.copy()
    kids = df["children"].fillna(0) + df["babies"]
    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["total_guests"] = df["adults"] + kids
    df["has_kids"] = (kids > 0).astype(int)
    df["arrival_weekday"] = arrival_date(df).dt.day_name()
    df["via_agent"] = df["agent"].notna().astype(int)
    df["via_company"] = df["company"].notna().astype(int)
    df["adr_per_person"] = df["adr"] / df["total_guests"].clip(lower=1)
    prev = df["previous_cancellations"] + df["previous_bookings_not_canceled"]
    df["prev_cancel_ratio"] = (df["previous_cancellations"] / prev).where(prev > 0, 0)
    return df


def drop_leaky(df, cols=LEAKY_COLS):
    """누수 열을 제거한 복사본을 반환한다."""
    return df.drop(columns=[c for c in cols if c in df.columns])


def final_features(df):
    """2일차에 확정한 최종 피처: 파생 피처 추가 → 누수·예약 후 기록·불안정 날짜 열 제거.

    파생 피처가 도착일(연도 포함)을 쓰므로 반드시 추가를 먼저 한다.
    """
    return drop_leaky(add_features(df), LEAKY_COLS + POST_BOOKING_COLS + UNSTABLE_DATE_COLS)


def split_xy(df, target=TARGET):
    return df.drop(columns=target), df[target]
