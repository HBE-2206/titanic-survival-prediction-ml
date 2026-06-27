import os
import pickle
import pandas as pd
import streamlit as st

# --- Page Config ---
st.set_page_config(
    page_title="Titanic Predictor | Slate Pro",
    page_icon="🚢",
    layout="wide"
)

# --- Theme Implementation ---
# Premium Dark Theme Colors
bg_color = "#0F1117"
card_color = "#1A1C24"
button_color = "#3B82F6"
button_hover = "#2563EB"
text_color = "#F8FAFC"
muted_text = "#94A3B8"
border_color = "#2E313D"
success_color = "#10B981"
success_bg = "rgba(16, 185, 129, 0.1)"
error_color = "#EF4444"
error_bg = "rgba(239, 68, 68, 0.1)"

# Custom CSS Inject
st.markdown(f"""
    <style>
    /* Main Background */
    .stApp {{
        background-color: {bg_color};
        color: {text_color};
    }}
    
    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {card_color};
        border-right: 1px solid {border_color};
    }}

    /* Card/Metric Styling */
    div[data-testid="metric-container"] {{
        background-color: {card_color};
        border: 1px solid {border_color};
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }}
    
    /* Text colors */
    [data-testid="stMetricValue"] {{
        color: {text_color} !important;
        font-weight: 700 !important;
    }}
    [data-testid="stMetricLabel"] {{
        color: {muted_text} !important;
        font-weight: 600 !important;
    }}

    /* Professional Button */
    .stButton>button {{
        width: 100%;
        border-radius: 8px;
        height: 3.5em;
        background: linear-gradient(135deg, {button_color}, #1D4ED8);
        color: white;
        border: none;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 10px rgba(59, 130, 246, 0.3);
    }}
    
    .stButton>button:hover {{
        background: linear-gradient(135deg, {button_hover}, #1E3A8A);
        box-shadow: 0 6px 15px rgba(59, 130, 246, 0.4);
        transform: translateY(-2px);
    }}

    /* Custom Success/Error boxes */
    .prediction-survived {{
        padding: 25px;
        background-color: {success_bg};
        border: 1px solid {success_color};
        border-radius: 12px;
        color: {success_color};
        text-align: center;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.1);
        backdrop-filter: blur(10px);
    }}
    
    .prediction-died {{
        padding: 25px;
        background-color: {error_bg};
        border: 1px solid {error_color};
        border-radius: 12px;
        color: {error_color};
        text-align: center;
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.1);
        backdrop-filter: blur(10px);
    }}
    
    .prediction-survived h2, .prediction-died h2 {{
        margin-top: 0;
        margin-bottom: 10px;
        font-weight: 700;
    }}
    
    .prediction-survived p, .prediction-died p {{
        margin-bottom: 0;
        font-size: 1.1em;
        font-weight: 500;
    }}

    hr {{ border-color: {border_color}; opacity: 0.5; }}
    </style>
""", unsafe_allow_html=True)


# --- Functions ---
@st.cache_resource
def load_model():
    """Auto-load trained ML model."""
    paths = [r'd:\ML NTI MATERYAL\pro ml\titanic_model.pkl', 'titanic_model.pkl']
    for path in paths:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
    return None


model = load_model()

# --- APP INTERFACE ---
st.title("🚢 Titanic Survival Predictor")
st.markdown(f"<p style='color: {muted_text};'>Advanced Passenger Risk Assessment Tool</p>", unsafe_allow_html=True)
st.divider()

# --- INPUT SECTION ---
with st.sidebar:
    st.markdown(f"### <span style='color: {text_color}'>Passenger Data</span>", unsafe_allow_html=True)
    pclass = st.selectbox('Ticket Class', [1, 2, 3], index=2)
    sex = st.radio('Gender', ['Male', 'Female'], horizontal=True)
    age = st.slider('Age', 0, 80, 25)
    sibsp = st.number_input('Siblings/Spouses', 0, 10, 0)
    parch = st.number_input('Parents/Children', 0, 10, 0)
    fare = st.number_input('Fare ($)', 0.0, 500.0, 32.0)
    embarked = st.selectbox('Embarkation Port', ['Cherbourg', 'Queenstown', 'Southampton'])

# Prepare Data for Model
sex_val = 1 if sex == 'Male' else 0
emb_dict = {'Cherbourg': 0, 'Queenstown': 1, 'Southampton': 2}
input_data = pd.DataFrame({
    'Pclass': [pclass], 
    'Sex': [sex_val], 
    'Age': [float(age)],
    'Sibsp': [sibsp], 
    'Parch': [parch], 
    'Fare': [float(fare)],
    'Embarked': [emb_dict[embarked]]
})

# --- RESULTS SECTION ---
st.subheader("📋 Current Selection")
cols = st.columns(4)
cols[0].metric("Class", f"P{pclass}")
cols[1].metric("Gender", sex)
cols[2].metric("Age", f"{age}y")
cols[3].metric("Fare", f"${fare:.1f}")

st.markdown("<br>", unsafe_allow_html=True)

# Predict Button and Output
b_col1, b_col2, b_col3 = st.columns([1, 1.5, 1])
with b_col2:
    if st.button('RUN PREDICTION MODEL'):
        if model:
            prediction = model.predict(input_data)
            prob = model.predict_proba(input_data)
            
            st.markdown("<br>", unsafe_allow_html=True)
            if prediction[0] == 1:
                st.balloons()
                st.markdown(f"""
                <div class="prediction-survived">
                    <h2>✨ Result: SURVIVED</h2>
                    <p>Confidence Level: {prob[0][1]:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-died">
                    <h2>⚠️ Result: DID NOT SURVIVE</h2>
                    <p>Fatality Probability: {prob[0][0]:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error("Model weights (titanic_model.pkl) not found. Please train the model first.")

st.markdown(
    f"<br><br><center style='color: {muted_text}; font-size: 0.8em;'>"
    "Powered by Scikit-Learn | Slate Pro Theme"
    "</center>", 
    unsafe_allow_html=True
)