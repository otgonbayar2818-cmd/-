# 기대수명 예측 — 다중 특성 회귀 파이프라인

WHO Life Expectancy 데이터를 이용한 다항 회귀 모델 파이프라인 및 Streamlit 웹 서비스

---

## 파일 구성

| 파일 | 설명 |
|------|------|
| `train_models.ipynb` | 데이터 로드 → 모델 학습 → 저장 → Streamlit 실행 (Colab용) |
| `app.py` | Streamlit 웹 대시보드 |
| `model_linear.pkl` | 1차 선형 회귀 파이프라인 |
| `model_poly.pkl` | 3차 다항 회귀 파이프라인 (과대적합 유도) |
| `model_ridge.pkl` | 3차 다항 + Ridge 규제 파이프라인 |
| `feature_info.pkl` | 슬라이더 범위 메타데이터 |
| `performance.csv` | 모델 성능 비교 테이블 |

---

## 사용 방법

### Google Colab에서 실행

1. `train_models.ipynb` 를 Colab에서 열기
2. 셀을 **위에서 아래로 순서대로** 실행
3. ngrok 토큰 입력 후 마지막 셀 실행
4. 출력된 `https://xxxx.ngrok-free.app` URL을 모바일로 접속

### ngrok 토큰 발급

[https://dashboard.ngrok.com](https://dashboard.ngrok.com) → 회원가입 → Your Authtoken 복사

---

## 선택 독립변수

- `Adult Mortality` — 성인 사망률
- `BMI` — 체질량지수
- `GDP` — 국내총생산

*(Schooling은 과제 조건에 따라 제외)*

---

## 모델 비교

| 모델 | 설명 | 특징 |
|------|------|------|
| Linear | 1차 선형 회귀 | 기준 모델 |
| Poly | 3차 다항 회귀 | 과대적합 유도 (Train R²↑ / Test R²↓) |
| Ridge | 3차 다항 + 규제 | 최적 일반화 성능 |
