import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(page_title="기대수명 예측 대시보드", layout="wide")

# ── 모델 및 메타데이터 로드 ───────────────────────────
@st.cache_resource
def load_models():
    return {
        "Linear": joblib.load("model_linear.pkl"),
        "Poly":   joblib.load("model_poly.pkl"),
        "Ridge":  joblib.load("model_ridge.pkl"),
    }

@st.cache_resource
def load_feature_info():
    return joblib.load("feature_info.pkl")

models    = load_models()
feat_info = load_feature_info()
FEATURES  = feat_info["features"]
RANGES    = feat_info["ranges"]
MEANS     = feat_info["means"]

# ── 사이드바 ─────────────────────────────────────────
st.sidebar.title("⚙️ 입력값 설정")
st.sidebar.markdown("---")

user_input = {}
for feat in FEATURES:
    mn, mx = RANGES[feat]
    mean   = MEANS[feat]
    step   = round((mx - mn) / 100, 2)
    user_input[feat] = st.sidebar.slider(
        label     = feat,
        min_value = float(mn),
        max_value = float(mx),
        value     = float(mean),
        step      = step,
    )

st.sidebar.markdown("---")
model_name = st.sidebar.selectbox(
    "모델 선택",
    ["Linear", "Poly", "Ridge"],
    help="Linear=선형, Poly=다항(과대적합), Ridge=규제 적용"
)

# ── 메인 대시보드 ─────────────────────────────────────
st.title("🌍 기대수명 예측 대시보드")
st.caption("WHO Life Expectancy 데이터 기반 · 다중 특성 다항 회귀 파이프라인")
st.markdown("---")

# 예측 실행
X_input = pd.DataFrame([user_input])
pred    = float(models[model_name].predict(X_input)[0])
pred    = round(max(0.0, min(pred, 100.0)), 2)

# 예측 결과 출력 (큰 글씨)
c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    st.metric("예측 기대수명", f"{pred} 세", f"{model_name} 모델")
with c2:
    st.metric("선택 모델", model_name)
with c3:
    st.metric("입력 특성 수", len(FEATURES))

st.markdown("---")

# ── 성능 비교 섹션 ────────────────────────────────────
st.subheader("📊 모델 성능 비교")

try:
    perf_df = pd.read_csv("performance.csv")

    # 테이블
    st.dataframe(perf_df, use_container_width=True)

    # 막대그래프
    fig, ax = plt.subplots(figsize=(6, 3.5))
    colors  = ["#378ADD", "#E24B4A", "#3B6D11"]
    bars    = ax.bar(perf_df["Model"], perf_df["Test R²"],
                     color=colors, width=0.45, edgecolor="none")
    ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=11)
    ax.set_title("Test R² Score — Model Comparison", fontsize=13, pad=10)
    ax.set_ylabel("R² Score")
    ax.set_ylim(min(perf_df["Test R²"].min() - 0.15, -0.2), 1.1)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

except FileNotFoundError:
    st.warning("⚠️ performance.csv 없음 → train_models.ipynb를 먼저 실행하세요.")

st.markdown("---")
st.caption("선택 독립변수: " + " · ".join(FEATURES))
