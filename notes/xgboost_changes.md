# XGBoost 추가 — 수정 내역 (2026-11-06)

2일차 노트북에 "XGBoost는 미설치라 LightGBM·CatBoost로 대체"라고 적혀 있던 부분을 실제 XGBoost 비교로 바꾸고,
3일차 노트북에 "XGBoost로 바꿔도 결론이 같은가" 장을 추가했다. 커밋은 아직 안 함.

## 0. 환경

- `.venv`에 `xgboost 3.4.1` 설치 (`.venv\Scripts\python.exe -m pip install xgboost`)
- 두 노트북 모두 `.venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace ...`로 재실행
- 커널이 `.venv\Scripts\python.exe`로 실행되는 것 확인
- 기존 셀들의 출력(숫자)은 수정 전과 동일하게 재현됨

---

## 1. `notebooks/02_validation_models.ipynb` (2일차)

> 2026-11-06 17:20에 XGBoost를 넣기 전 버전에 메모 셀 18개가 추가된 상태로 저장되면서 XGBoost 수정이 덮어써졌다. 메모가 추가된 현재 파일(셀 75개) 위에 아래 수정을 다시 넣었다. 셀 번호는 현재 파일 기준.
> 셀 59 메모의 "총 8종의 모델 후보군"은 XGBoost 2개가 추가돼 이제 10종이다 (직접 쓴 메모라 수정하지 않음).

**셀 1 — import**
```diff
 from sklearn.model_selection import GroupShuffleSplit, train_test_split
+from xgboost import XGBClassifier
```

**셀 4 — 모델 헬퍼** (`lgbm()` 바로 아래)
```diff
+def xgb(**kw): # LightGBM과 같은 부스팅 계열. 비교를 위해 트리 수·학습률을 LightGBM과 맞춤
+    params = dict(n_estimators=500, learning_rate=0.05, max_depth=6, tree_method="hist", n_jobs=-1, random_state=SEED)
+    params.update(kw)
+    return XGBClassifier(**params)
```

**셀 55 — 5장 설명 마크다운**
```diff
-- XGBoost는 설치돼 있지 않아 같은 부스팅 계열인 LightGBM, CatBoost로 비교한다.
+- 부스팅 계열은 LightGBM, XGBoost, CatBoost 세 가지를 비교한다.
```

**셀 58 — 모델 비교 `candidates`** (CatBoost 바로 위)
```diff
+    "XGBoost (max_depth=6)": (xgb(), True),
+    "XGBoost (max_depth=4)": (xgb(max_depth=4), True),
```

**셀 60 — 결론 마크다운**
```diff
-- 부스팅 계열(LightGBM, CatBoost)이 RandomForest·로지스틱 회귀보다 확실히 낫다.
+- 부스팅 계열(LightGBM, XGBoost, CatBoost)이 RandomForest·로지스틱 회귀보다 확실히 낫다.
+- XGBoost(`max_depth=6`)도 0.9007로 같은 수준이다. 부스팅 세 가지가 0.900~0.902 안에 모여 있어 **알고리즘보다 피처·평가 방식이 성능을 좌우한다**고 볼 수 있다. 셋이 사실상 동률이므로 빠르고 단순한 LightGBM을 유지한다.
```

**결과 (검증 기간)**

| 모델 | ROC-AUC | PR-AUC | 학습 시간 |
|---|---|---|---|
| LightGBM (31, subsample 0.8) | 0.9018 | 0.8571 | 1.9s |
| CatBoost (depth=6) | 0.9015 | 0.8553 | 11.8s |
| LightGBM (num_leaves=31) — 최종 모델 | 0.9011 | 0.8564 | 2.0s |
| XGBoost (max_depth=6) | 0.9007 | 0.8558 | 2.4s |
| XGBoost (max_depth=4) | 0.8993 | 0.8523 | 2.0s |

→ 최종 모델은 LightGBM 그대로.

**셀 64 — 6장 테스트 최종 평가 `final_specs`** (비교 대상으로 추가)
```diff
     "LightGBM (최종 피처)": (X, lgbm(num_leaves=31), True),
+    "XGBoost (최종 피처)": (X, xgb(), True),
```
- 셀 66 ROC·PR 곡선은 `final_models`를 돌며 그리므로 XGBoost 곡선이 자동으로 추가됨
- 테스트 결과: XGBoost ROC-AUC 0.869 / PR-AUC 0.825 / F1 0.669 (LightGBM 0.870 / 0.828 / 0.680)

**셀 70 — 결과 해석 마크다운** ("같은 정직한 피처끼리 비교하면..." 줄 아래)
```diff
+  XGBoost(최종 피처)도 0.869로 LightGBM과 같은 수준이다. 검증 기간의 동률이 테스트에서도 그대로 이어져, 부스팅 모델 중 무엇을 골랐는지는 결과에 거의 영향이 없다.
```
**셀 73 — 2일차 정리 출력**
```diff
-for name in ["1일차 LogReg (기준점)", "LogReg (최종 피처)", "LightGBM (최종 피처)"]:
+for name in ["1일차 LogReg (기준점)", "LogReg (최종 피처)", "LightGBM (최종 피처)", "XGBoost (최종 피처)"]:
```
- 셀 65 메모(최종 후보 목록 표)는 직접 쓴 메모라 수정하지 않음 — XGBoost 행 추가가 필요하면 직접

---

## 2. `notebooks/03_interpretation_threshold.ipynb` (3일차)

**새 5장 "XGBoost로 바꿔도 결론이 같은가"** — 4장(건수 예측)과 한계 장 사이에 셀 5개 추가
1. 마크다운: 목적 설명 (1~4장 결론이 LightGBM 특유의 결과인지 확인)
2. 코드: XGBoost 선택용(학습 구간)·최종(학습+검증) 모델 → 성능·보정·threshold·일별 편향 비교표 (`robust`)
3. 마크다운: SHAP 비교 안내
4. 코드: 같은 5,000건으로 SHAP 비중 비교 (`shap_compare`)
5. 마크다운: 결론

| | LightGBM | XGBoost |
|---|---|---|
| 테스트 ROC-AUC / PR-AUC | 0.870 / 0.828 | 0.869 / 0.825 |
| 평균 예측 확률 (실제 0.405) | 0.339 | 0.341 |
| threshold A / B (검증에서 선택) | 0.10 / 0.85 | 0.09 / 0.84 |
| 테스트 예약당 비용 A / B | 0.367 / 0.309 | 0.365 / 0.311 |
| 일별 취소 건수 편향 | −6.0건 | −5.8건 |

SHAP 상위 6개 피처(`deposit_type`, `country`, `market_segment`, `total_of_special_requests`, `lead_time`, `agent`)는 순서만 조금 다르고 같음.
XGBoost는 `deposit_type` 25% → 20%, `agent` 7% → 11%.

결론: 3일차 결론은 모델 선택과 무관. `deposit_type`·`country` 의존과 확률 과소 추정은 알고리즘을 바꿔서는 해결되지 않는 한계.

**그 밖의 수정**
- 셀 0 목표: 5번 "XGBoost로 바꿔도 결론이 같은지" 추가, 한계는 6번으로
- 셀 1 import: `XGBClassifier`, `roc_auc_score`, `average_precision_score` 추가
- 장 번호: "한계와 개선 방향" 5장 → 6장, "3일차 정리" 6장 → 7장, 4장 끝 "(→ 5장)" → "(→ 6장)"
- 2장 마크다운 "5장 건수 예측" → "4장 건수 예측" (원래부터 틀려 있던 번호)
- 마지막 정리 셀: `XGBoost로 바꿔도: 테스트 ROC-AUC 0.869, threshold A 0.09 / B 0.84 → 결론 동일` 출력 줄 추가

---

## 3. `README.md`

- 19줄 설치 목록: `lightgbm, catboost` → `lightgbm, xgboost, catboost`
- 40줄 폴더 구조: 3일차 설명에 "XGBoost 비교" 추가
- 4장 검증 비교표: `XGBoost (max_depth=6) | 0.901 | 0.856 | 2s` 행 추가
- 4장 결론: "XGBoost는 미설치, 같은 부스팅 계열 둘로 대체" → "LightGBM·XGBoost·CatBoost 부스팅 3종이 0.900~0.902로 동률 → 가장 빠르고 단순한 LightGBM 선택"
- 4장 테스트 기간 표: `XGBoost (최종 피처) | 0.869 | 0.825 | 0.797 | 0.576 | 0.669` 행 추가
- 6장 끝(7장 바로 앞): "XGBoost로 바꿔도 같은가 (견고성 확인)" 소절 추가 (위 표 + 결론 2줄)

※ README는 이번 작업 전에도 수정 중이던 상태라 `git diff README.md`에는 이전 작업분도 함께 보임.

## 남은 확인 사항

- README 149줄 "→ 5장에서 비용 기준으로 다시 정함": README 구조상 threshold는 6장이라 번호가 틀려 보임 (원래부터 있던 문제, 아직 안 고침).
