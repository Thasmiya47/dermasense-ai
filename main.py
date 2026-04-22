import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# ------------------ CONFIG ------------------
st.set_page_config(page_title="DermaSense AI", layout="centered")

if "started" not in st.session_state:
    st.session_state.started = False

def go_home():
    st.session_state.started = False
    st.rerun()

# ------------------ THEME ------------------
theme = st.sidebar.radio("🎨 Theme", ["Dark", "Light"])
dark = theme == "Dark"

bg = "#0f0c29" if dark else "#ffffff"
text = "white" if dark else "#111111"

# ------------------ CSS ------------------
st.markdown(f"""
<style>

.stApp {{
    background-color: {bg};
    color: {text};
}}

html, body, [class*="css"] {{
    color: {text} !important;
}}

h1,h2,h3,h4,h5,h6,p,label {{
    color: {text} !important;
}}

.title {{
    text-align: center;
    font-size: 52px;
    font-weight: bold;
    color: #b388ff;
}}

.card {{
    background: linear-gradient(135deg, #2b1055, #7597de);
    padding: 14px;
    border-radius: 14px;
    color: white;
    margin: 10px 0;
}}

.glow-card {{
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.2);
    padding: 12px;
    border-radius: 12px;
    margin: 8px 0;
    backdrop-filter: blur(10px);
}}

.stButton>button {{
    width: 100%;
    padding: 0.7rem;
    font-size: 18px;
    border-radius: 12px;
    font-weight: bold;
    border: none;
    transition: 0.3s;
    background: linear-gradient(90deg,#8e2de2,#4a00e0);
    color:white;
    box-shadow:0 0 20px #8e2de2;
}}

</style>
""", unsafe_allow_html=True)

# ------------------ DATA ------------------
data = pd.DataFrame({
    "skin_type":[0,1,2,0,1,2,0,1,2,1,2,0,1,2,0,1,2,0,1,2],
    "acne":[0,1,2,1,0,2,2,1,0,0,1,2,1,0,2,1,0,2,1,0],
    "pigmentation":[0,1,0,1,0,1,0,1,0,1,0,1,1,0,1,0,1,0,1,0],
    "sensitivity":[0,0,1,1,0,1,0,1,1,0,0,1,1,0,1,0,1,0,0,1],
    "label":[1,0,2,0,1,2,2,0,1,1,0,2,0,1,2,1,0,2,1,0]
})

X = data.drop("label", axis=1)
y = data["label"]

model = LogisticRegression(max_iter=500)
model.fit(X, y)

labels = {
    0: "Acne-Prone Skin 🔴",
    1: "Normal Skin 🟢",
    2: "Sensitive Skin 🟡"
}

mapping_skin = {"oily":0,"dry":1,"combination":2}
mapping_acne = {"low":0,"medium":1,"high":2}
mapping_yesno = {"no":0,"yes":1}

def get_days(label):
    return 28 if label == 0 else 14 if label == 1 else 18

# ------------------ RECOMMENDATIONS ------------------
recommendations = {
    0: {
        "morning": ["Salicylic Acid Cleanser", "Niacinamide Serum 10%", "Oil-free Moisturizer", "SPF 50+"],
        "night": ["Benzoyl Peroxide 2.5%", "Retinol (2–3x/week)", "Gel Moisturizer"],
        "tips": ["Avoid oily food", "Clean pillow covers", "Hydration"]
    },
    1: {
        "morning": ["Vitamin C Serum 15%", "Hyaluronic Acid 1%", "Moisturizer", "SPF 30–50"],
        "night": ["Hydrating Cleanser", "Peptide Cream", "Light Moisturizer"],
        "tips": ["Balanced diet", "Sleep 7–8 hours", "Hydration"]
    },
    2: {
        "morning": ["Centella Serum 5%", "Ceramide Moisturizer 10%", "Sunscreen SPF 50+"],
        "night": ["Aloe Vera Gel 90%", "Barrier Cream", "Hydration Serum"],
        "tips": ["Avoid harsh products", "Patch test", "Gentle care"]
    }
}

# ------------------ HOME ------------------
if not st.session_state.started:

    st.markdown('<div class="title">🧴 DermaSense AI</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="glow-card" style="text-align:center; line-height:2;">
        🧠 Detects skin type using AI<br>
        📊 Analyzes acne, pigmentation & sensitivity<br>
        💡 Personalized skincare routine<br>
        🌙 Morning & Night care plan<br>
        📈 Recovery forecast prediction
    </div>
    """, unsafe_allow_html=True)

    if st.button("🔮 Analyze Skin"):
        st.session_state.started = True
        st.rerun()

# ------------------ MAIN ------------------
else:

    st.subheader("Skin Analysis Dashboard")

    skin_type = st.selectbox("Skin Type", ["Oily","Dry","Combination"]).lower()
    acne = st.selectbox("Acne Level", ["Low","Medium","High"]).lower()
    pigmentation = st.selectbox("Pigmentation", ["No","Yes"]).lower()
    sensitivity = st.selectbox("Sensitivity", ["No","Yes"]).lower()

    if st.button("🔮 ANALYZE SKIN"):

        input_data = pd.DataFrame([[
            mapping_skin[skin_type],
            mapping_acne[acne],
            mapping_yesno[pigmentation],
            mapping_yesno[sensitivity]
        ]], columns=["skin_type","acne","pigmentation","sensitivity"])

        prediction = model.predict(input_data)[0]
        confidence = np.max(model.predict_proba(input_data)) * 100

        rec = recommendations[prediction]

        tab1, tab2, tab3 = st.tabs(["🧬 Skin Type","💡 Recommendation","📊 Recovery"])

        # ---------------- TAB 1 (UPDATED) ----------------
        with tab1:
            st.markdown(f"""
            <div class="card">
                <h2>{labels[prediction]}</h2>
                AI Confidence: {confidence:.2f}%
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### Selected Inputs")

            st.markdown(f"""
            <div class="glow-card">
                ✔ Skin Type: {skin_type.title()}<br>
                ✔ Acne Level: {acne.title()}<br>
                ✔ Pigmentation: {pigmentation.title()}<br>
                ✔ Sensitivity: {sensitivity.title()}
            </div>
            """, unsafe_allow_html=True)

        # ---------------- TAB 2 (UPDATED) ----------------
        with tab2:

            st.markdown("### 🌞 Morning Routine")
            for i in rec["morning"]:
                st.markdown(f"<div class='glow-card'>✔ {i}</div>", unsafe_allow_html=True)

            st.markdown("### 🌙 Night Routine")
            for i in rec["night"]:
                st.markdown(f"<div class='glow-card'>✔ {i}</div>", unsafe_allow_html=True)

            st.markdown("### 💡 Tips")
            for i in rec["tips"]:
                st.markdown(f"<div class='glow-card'>✔ {i}</div>", unsafe_allow_html=True)

        # ---------------- TAB 3 ----------------
        with tab3:

            days = get_days(prediction)

            st.markdown(f"""
            <div class="card">
                <h3>Recovery Forecast</h3>
                <h2>{days} Days</h2>
            </div>
            """, unsafe_allow_html=True)

            x = np.arange(1, days + 1)
            y = np.log1p(x) / np.log1p(days) * 100

            fig, ax = plt.subplots()
            ax.plot(x, y, linewidth=3)

            ax.set_xlabel("Days")
            ax.set_ylabel("Improvement %")
            ax.set_title("Skin Recovery Progress")
            ax.set_ylim(0, 100)
            ax.grid(True)

            if dark:
                ax.set_facecolor("white")
                fig.patch.set_facecolor("white")
                ax.tick_params(colors="black")
            else:
                ax.set_facecolor("#0f0c29")
                fig.patch.set_facecolor("#0f0c29")
                ax.tick_params(colors="white")

            st.pyplot(fig)
