import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# ---- 설정 ----
st.set_page_config(page_title="망막 손상 위험도 예측기", page_icon="🧠", layout="centered")

# ---- 제목 ----
st.title("📱 망막 손상 위험도 예측기")
st.markdown("### 하루 핸드폰 사용 시간이 **눈 건강에 어떤 영향을 줄까?**")
st.markdown("---")

# ---- 데이터 불러오기 및 모델 학습 ----
df = pd.read_csv('retina_data.csv')
X = df[['시간_분']]
y = df['망막 손상 위험도']

model = LinearRegression()
model.fit(X, y)

# ---- 사용자 입력 ----
st.sidebar.header("🔢 사용 시간 입력")
user_time = st.sidebar.number_input("하루 핸드폰 사용 시간 (분)", min_value=0, max_value=1440, step=10)

# ---- 예측 결과 ----
if user_time:
    prediction = round(model.predict([[user_time]])[0], 2)

    st.subheader("🔍 예측 결과")
    st.markdown(f"**사용 시간:** `{user_time}분`")
    st.markdown(f"**예상 망막 손상 위험도:** `{prediction} / 10`")

    # 상태에 따른 메시지 출력
    st.markdown("---")
    if prediction >= 8:
        st.error("🚨 매우 높은 위험도! 휴대폰 사용 시간을 줄이세요!")
        st.image("https://cdn-icons-png.flaticon.com/512/564/564619.png", width=100)
    elif prediction >= 6:
        st.warning("⚠️ 주의가 필요한 수준입니다.")
    else:
        st.success("✅ 비교적 안전한 수준입니다.")

# ---- 참고 정보 ----
with st.expander("📘 참고: 망막 손상 데이터 설명"):
    st.dataframe(df, use_container_width=True)
