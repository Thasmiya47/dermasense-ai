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

bg = "linear-gradient(135deg, #0f0c29, #302b63, #24243e)" if dark else "white"
text = "white" if dark else "black"

st.markdown(f"""
<style>
.stApp {{
    background: {bg};
    color: {text};
}}

.title {{
    text-align: center;
    font-size: 52px;
    font-weight: bold;
    color: #b388ff;
}}

.big-subtitle {{
    text-align: center;
    font-size: 26px;
    font-weight: bold;
    color: #d1c4ff;
    margin-top: 10px;
}}

.desc {{
    text-align: center;
    font-size: 16px;
    margin-top: 10px;
    line-height: 1.8;
}}

.card {{
    background: linear-gradient(135deg, #2b1055, #7597de);
    padding: 14px;
    border-radius: 14px;
    margin-top: 8px;
    color: white;
    box-shadow: 0 0 15px #6a11cb;
}}

.small-card {{
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    padding: 10px;
    border-radius: 12px;
    margin-top: 6px;
    color: white;
    box-shadow: 0 0 10px #4a4aff;
}}

</style>
""", unsafe_allow_html=True)

# ------------------ DATA ------------------
data = pd.DataFrame({
    "skin_type": [0,1,2,0,1,2,0,1,2,1,2,0,1,2,0,1,2,0,1,2],
    "acne":      [0,1,2,1,0,2,2,1,0,0,1,2,1,0,2,1,0,2,1,0],
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
    return 28 if label == 0 else 10 if label == 1 else 18

recommendations = {
    0: {
        "morning": [
            "Salicylic Acid Cleanser (0.5% - 2%)",
            "Niacinamide Serum (5% - 10%)",
            "Oil-free Moisturizer",
            "SPF 50 Sunscreen"
        ],
        "night": [
            "Benzoyl Peroxide (2.5% - 5%)",
            "Retinol (0.1% - 0.3%) - 2 to 3 times/week",
            "Gel-based Moisturizer"
        ],
        "tips": [
            "Avoid oily and junk food",
            "Clean pillow covers regularly",
            "Stay hydrated (2–3L water daily)"
        ]
    },

    1: {
        "morning": [
            "Vitamin C Serum (10% - 15%)",
            "Hyaluronic Acid (1% - 2%)",
            "Moisturizer with ceramides",
            "SPF 30–50 Sunscreen"
        ],
        "night": [
            "Hydrating Cleanser",
            "Peptide Moisturizer",
            "Light Repair Cream"
        ],
        "tips": [
            "Maintain balanced diet",
            "Sleep 7–8 hours",
            "Drink enough water"
        ]
    },

    2: {
        "morning": [
            "Centella Asiatica Serum",
            "Ceramide Moisturizer",
            "Mineral Sunscreen (SPF 30+)"
        ],
        "night": [
            "Soothing Gel (Aloe/Cica based)",
            "Barrier Repair Cream",
            "Hydrating Serum"
        ],
        "tips": [
            "Avoid harsh exfoliation",
            "Do patch testing before products",
            "Use gentle skincare only"
        ]
    }
}

# ------------------ HOME PAGE ------------------
if not st.session_state.started:

    st.markdown('<div class="title">🧴 DermaSense AI</div>', unsafe_allow_html=True)

    # ✔ BIG SUBTITLE
    st.markdown("""
    <div class="big-subtitle">
        AI-driven Skin Analysis & Recommendation System
    </div>
    """, unsafe_allow_html=True)

    # ✔ DESCRIPTION (5 POINTS ROW STYLE)
    st.markdown("""
    <div class="desc">
        🧠 Detects your skin type using AI<br>
        📊 Analyzes acne, pigmentation & sensitivity<br>
        💡 Gives personalized skincare routine<br>
        🌙 Suggests morning & night care steps<br>
        📈 Shows recovery improvement forecast
    </div>
    """, unsafe_allow_html=True)

    if st.button("🔮 Analyze Skin"):
        st.session_state.started = True
        st.rerun()

# ------------------ MAIN APP ------------------
else:

    col1, col2 = st.columns([8,2])
    with col1:
        st.subheader("Skin Analysis Dashboard")
    with col2:
        st.button("🏠 Home", on_click=go_home)

    skin_type = st.selectbox("Skin Type", ["Oily","Dry","Combination"]).lower()
    acne = st.selectbox("Acne Level", ["Low","Medium","High"]).lower()
    pigmentation = st.selectbox("Pigmentation", ["No","Yes"]).lower()
    sensitivity = st.selectbox("Sensitivity", ["No","Yes"]).lower()

    if st.button("🔮 Analyze"):

        input_data = pd.DataFrame([[
            mapping_skin[skin_type],
            mapping_acne[acne],
            mapping_yesno[pigmentation],
            mapping_yesno[sensitivity]
        ]], columns=["skin_type","acne","pigmentation","sensitivity"])

        prediction = model.predict(input_data)[0]
        probs = model.predict_proba(input_data)[0]
        confidence = np.max(probs) * 100

        rec = recommendations[prediction]

        tab1, tab2, tab3 = st.tabs(["🧬 Skin Type","💡 Recommendation","📊 Recovery"])

        # ------------------ TAB 1 ------------------
        with tab1:
            st.markdown(f"""
            <div class="card">
                <h2>{labels[prediction]}</h2>
                🧠 AI Confidence: {confidence:.2f}%
            </div>
            """, unsafe_allow_html=True)

            # ✔ SELECTED INPUTS BELOW SKIN TYPE
            st.markdown("### 🧾 Selected Inputs")

            st.markdown(f"""
            <div class="small-card">
                Skin Type: {skin_type.title()}<br>
                Acne Level: {acne.title()}<br>
                Pigmentation: {pigmentation.title()}<br>
                Sensitivity: {sensitivity.title()}
            </div>
            """, unsafe_allow_html=True)

        # ------------------ TAB 2 ------------------
        with tab2:

            st.markdown("### 🌞 Morning Routine")
            for i in rec["morning"]:
                st.markdown(f"<div class='small-card'>✔ {i}</div>", unsafe_allow_html=True)

            st.markdown("### 🌙 Night Routine")
            for i in rec["night"]:
                st.markdown(f"<div class='small-card'>✔ {i}</div>", unsafe_allow_html=True)

            st.markdown("### 💡 Tips")
            for i in rec["tips"]:
                st.markdown(f"<div class='small-card'>✔ {i}</div>", unsafe_allow_html=True)

        # ------------------ TAB 3 ------------------
        with tab3:

            days = get_days(prediction)

            st.markdown(f"""
            <div class="card">
                <h3>📊 Recovery Time</h3>
                <h2>{days} Days</h2>
            </div>
            """, unsafe_allow_html=True)

            x = np.arange(days)
            y = np.log1p(x) / np.log1p(days) * 100

            fig, ax = plt.subplots()
            ax.plot(x, y, linewidth=3)
            ax.set_ylim(0, 100)
            ax.set_title("Skin Improvement Over Time")
            ax.grid(True)

            st.pyplot(fig)