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

bg = "#0f0c29" if dark else "white"
text = "white" if dark else "black"

# ------------------ FIXED CSS ------------------
st.markdown(f"""
<style>

/* Main background */
.stApp {{
    background-color: {bg};
    color: {text};
}}

/* FORCE ALL TEXT */
html, body, [class*="css"] {{
    color: {text} !important;
}}

/* Headings */
h1, h2, h3, h4, h5, h6, p, label {{
    color: {text} !important;
}}

/* Title */
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
}}

.desc {{
    text-align: center;
    font-size: 16px;
    line-height: 1.8;
    color: {text};
}}

/* Cards */
.card {{
    background: linear-gradient(135deg, #2b1055, #7597de);
    padding: 14px;
    border-radius: 14px;
    margin-top: 8px;
    color: white;
}}

.small-card {{
    background: {("#1a1a2e" if dark else "#f2f2f2")};
    padding: 10px;
    border-radius: 12px;
    margin-top: 6px;
    color: {text};
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

# ------------------ RECOMMENDATIONS ------------------
recommendations = {
    0: {
        "morning": ["Salicylic Acid Cleanser", "Niacinamide Serum", "Oil-free Moisturizer", "SPF 50"],
        "night": ["Benzoyl Peroxide", "Retinol 2-3x/week", "Gel Moisturizer"],
        "tips": ["Avoid oily food", "Clean pillow covers", "Hydration"]
    },
    1: {
        "morning": ["Vitamin C Serum", "Hyaluronic Acid", "Moisturizer", "SPF 30-50"],
        "night": ["Hydrating Cleanser", "Peptide Cream", "Light Moisturizer"],
        "tips": ["Balanced diet", "Sleep 7-8 hours", "Hydration"]
    },
    2: {
        "morning": ["Centella Serum", "Ceramide Moisturizer", "Sunscreen"],
        "night": ["Aloe Gel", "Barrier Cream", "Hydration Serum"],
        "tips": ["Avoid harsh products", "Patch test", "Gentle care"]
    }
}

# ------------------ HOME ------------------
if not st.session_state.started:

    st.markdown('<div class="title">🧴 DermaSense AI</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="big-subtitle">
        AI-driven Skin Analysis & Recommendation System
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="desc">
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

        # ---------------- TAB 1 ----------------
        with tab1:
            st.markdown(f"""
            <div class="card">
                <h2>{labels[prediction]}</h2>
                🧠 AI Confidence: {confidence:.2f}%
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### 🧾 Selected Inputs")
            st.markdown(f"""
            <div class="small-card">
                Skin Type: {skin_type.title()}<br>
                Acne Level: {acne.title()}<br>
                Pigmentation: {pigmentation.title()}<br>
                Sensitivity: {sensitivity.title()}
            </div>
            """, unsafe_allow_html=True)

        # ---------------- TAB 2 ----------------
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

        # ---------------- TAB 3 ----------------
        with tab3:

            days = get_days(prediction)

            st.markdown(f"""
            <div class="card">
                <h3>📊 Recovery Forecast</h3>
                <h2>{days} Days</h2>
            </div>
            """, unsafe_allow_html=True)

            x = np.arange(1, days + 1)
            y = np.log1p(x) / np.log1p(days) * 100

            fig, ax = plt.subplots()
            ax.plot(x, y, linewidth=3)

            ax.set_xlabel("Days")
            ax.set_ylabel("Recovery / Improvement (%)")
            ax.set_title("Skin Improvement Over Time")
            ax.set_ylim(0, 100)
            ax.grid(True)

            if dark:
                ax.set_facecolor("#0f0c29")
                fig.patch.set_facecolor("#0f0c29")
                ax.tick_params(colors="white")
                ax.xaxis.label.set_color("white")
                ax.yaxis.label.set_color("white")
                ax.title.set_color("white")
            else:
                ax.set_facecolor("white")
                fig.patch.set_facecolor("white")
                ax.tick_params(colors="black")
                ax.xaxis.label.set_color("black")
                ax.yaxis.label.set_color("black")
                ax.title.set_color("black")

            st.pyplot(fig)
