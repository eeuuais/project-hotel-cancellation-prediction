"""전처리 파이프라인 구성 · 모델 평가용 함수 모음."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

# 숫자로 저장돼 있지만 의미상 범주(ID)인 열
ID_LIKE_COLS = ["agent", "company"]


def column_groups(X, id_like=ID_LIKE_COLS):
    """수치형 / 범주형 열 이름을 나눈다. ID 성격의 숫자 열은 범주형으로 보낸다."""
    cat = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c]) or c in id_like]
    num = [c for c in X.columns if c not in cat]
    return num, cat


def build_preprocessor(X, min_frequency=20):
    """수치형: 중앙값 대체 + 표준화 / 범주형: 'missing' 대체 + 원-핫(희귀 범주는 묶음)."""
    num, cat = column_groups(X)
    num_pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    cat_pipe = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="missing"),
        OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_frequency),
    )
    return ColumnTransformer([("num", num_pipe, num), ("cat", cat_pipe, cat)])


def to_str_ids(X, id_like=ID_LIKE_COLS):
    """ID 열을 문자열로 바꿔 원-핫 인코더가 범주로 다루게 한다. 결측은 그대로 둔다."""
    X = X.copy()
    for c in id_like:
        if c in X.columns:
            X[c] = X[c].astype("object").where(X[c].isna(), X[c].astype("Int64").astype(str))
    return X


def build_tree_preprocessor(X, min_frequency=20):
    """트리 모델용: 수치형은 중앙값 대체만 / 범주형은 정수 코드(희귀·처음 보는 범주는 묶음).

    트리는 크기 비교로 나누므로 표준화가 필요 없고, 원-핫보다 열이 적어 빠르다.
    """
    num, cat = column_groups(X)
    cat_pipe = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="missing"),
        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                       min_frequency=min_frequency),
    )
    # pandas로 출력해 학습·예측 모두 열 이름을 유지한다 (LightGBM 경고 방지, 중요도 해석용)
    return ColumnTransformer(
        [("num", SimpleImputer(strategy="median"), num), ("cat", cat_pipe, cat)],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def make_model(X, estimator, tree=False):
    """ID 열 문자열 변환 + 전처리 + 추정기를 하나의 파이프라인으로 묶는다.

    tree=True면 트리용 전처리(build_tree_preprocessor)를 쓴다.
    """
    prep = build_tree_preprocessor(X) if tree else build_preprocessor(X)
    return Pipeline([
        ("ids", FunctionTransformer(to_str_ids)),
        ("prep", prep),
        ("clf", estimator),
    ])


def evaluate(model, X, y, threshold=0.5):
    """분류 지표를 dict로 반환한다."""
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
    }


def compare(results):
    """{모델이름: evaluate 결과} → 비교표."""
    return pd.DataFrame(results).T.round(4)


def top_coefficients(model, n=15):
    """로지스틱 회귀 계수를 크기 순으로 정렬해 상위/하위 n개를 반환한다."""
    names = model.named_steps["prep"].get_feature_names_out()
    coef = pd.Series(model.named_steps["clf"].coef_[0], index=names).sort_values()
    return pd.concat([coef.head(n), coef.tail(n)])
