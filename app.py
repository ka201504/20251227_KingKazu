import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# 画面設定
st.set_page_config(layout="wide", page_title="ctDNA Resistance Predictor")

# --- タイトル ---
st.title("🧬 ctDNA-Based Resistance Simulator")
st.markdown("リキッドバイオプシーによる治療抵抗性の早期予測モデル")
st.write("---")

# --- 1. Patient Profile & Data Input (メイン画面上部に配置) ---
st.header("1. Patient Profile & Biomarkers")
input_col1, input_col2, input_col3 = st.columns(3)

with input_col1:
    in_ras = st.selectbox("RAS Status", ["Wild-type", "Mutant"])
    in_msi = st.selectbox("MSI Status", ["MSS", "MSI-H"])

with input_col2:
    in_nol3 = st.slider("NOL3 Expression Level", 0.0, 1.0, 0.2)
    st.caption("※資料に基づいた抵抗性因子(NOL3)")

with input_col3:
    ct_m0 = st.number_input("ctDNA Baseline (copy/mL)", 0, 1000, 500)
    ct_m3 = st.number_input("ctDNA Month 3 (copy/mL)", 0, 1000, 100)

st.write("---")

# --- 2. 模擬学習データの生成 (ML要素) ---
@st.cache_data
def train_mock_model():
    np.random.seed(42)
    n = 100
    X = pd.DataFrame({
        'ras': np.random.choice([0, 1], n),
        'msi': np.random.choice([0, 1], n),
        'nol3': np.random.rand(n),
        'ct_trend': np.random.rand(n)
    })
    # 教師データの作成ロジック（専門的背景を反映）
    y = X['ras'] * 0.5 + X['nol3'] * 0.4 + (1 - X['ct_trend']) * 0.3
    model = RandomForestRegressor(n_estimators=50).fit(X, y)
    return model, X.columns

model, features = train_mock_model()

# --- 3. 予測実行 ---
user_x = pd.DataFrame([[
    1 if in_ras == "Mutant" else 0,
    1 if in_msi == "MSI-H" else 0,
    in_nol3,
    (ct_m0 - ct_m3) / max(ct_m0, 1)
]], columns=features)

resistance_score = model.predict(user_x)[0]

# --- 4. 可視化 ---
st.header("2. Resistance Prediction & Simulation")
res_col1, res_col2 = st.columns([2, 1])

with res_col1:
    months = np.array([0, 3, 6, 9, 12])
    # 腫瘍量の推移シミュレーション
    trend = [ct_m0, ct_m3, ct_m3 * (1 + resistance_score), ct_m3 * (1 + resistance_score*3), ct_m3 * (1 + resistance_score*8)]
    chart_data = pd.DataFrame({"Month": months, "Predicted Tumor Burden (ctDNA)": trend})
    st.line_chart(chart_data, x="Month", y="Predicted Tumor Burden (ctDNA)")

with res_col2:
    st.metric("AI Resistance Score", f"{resistance_score:.2f}")
    st.progress(min(resistance_score, 1.0))
    st.write("AI判定：スコアが高いほど、早期の耐性クローン出現リスクを示唆します。")

# --- 5. 専門家へのディスカッション ---
st.write("---")
with st.expander("👨‍🔬 研究者（友人）への質問・ディスカッション項目"):
    st.markdown(f"""
    1. **MRD(微小残存病変)**: Month 3 で ctDNA が陽性( {ct_m3} )の場合、画像(CT)で再発が見える前に介入する意義をどう考える？
    2. **抵抗性因子**: 資料にあった **NOL3遺伝子** の発現が、抗EGFR薬の耐性を加速させる感覚は、研究データと乖離はないかな？
    3. **バイオダイナミクス**: ctDNA の再上昇（V字回復）の傾きに影響を与える因子として、他に何を組み込むべきだと思う？
    """)
