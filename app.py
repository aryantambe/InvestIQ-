import streamlit as st
import mysql.connector
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# --- Styling ---
st.markdown("""
    <style>
        body {
            background-color: #0e1117;
            color: white;
        }
        .main {
            background-color: #0e1117;
        }
        input, textarea, select {
            background-color: #1e2128 !important;
            color: white !important;
        }
        .stButton>button {
            background-color: #6c63ff;
            color: white;
            border-radius: 10px;
            padding: 0.6em 1.5em;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #a393eb;
        }
        .title-style {
            text-align: left;
            font-size: 3.5em;
            font-weight: bold;
            margin-bottom: 10px;
            animation: fadeIn 1.2s ease-in-out;
        }
        .fade-in {
            animation: fadeIn 1.5s ease-in-out;
        }
        .section-title {
            font-size: 1.4em;
            color: #ffffff;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
""", unsafe_allow_html=True)

# --- Load Dataset ---
df = pd.read_csv(r'C:\Users\DELL\OneDrive\Desktop\streamlit\DSBI_dataset.csv')
df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace(r'[^\w]', '', regex=True)
if 'Sr_No' in df.columns:
    df.drop(columns=['Sr_No'], inplace=True)
if df['Amount_in_USD'].dtype == 'object':
    df['Amount_in_USD'] = df['Amount_in_USD'].str.replace(',', '')
    df['Amount_in_USD'] = pd.to_numeric(df['Amount_in_USD'], errors='coerce')
df['City__Location'] = df['City__Location'].fillna("Unknown")
df['Industry_Vertical'] = df['Industry_Vertical'].fillna("Other")
df = df.dropna(subset=['Amount_in_USD', 'Investors_Name'])
df['Date_ddmmyyyy'] = pd.to_datetime(df['Date_ddmmyyyy'], dayfirst=True, errors='coerce')
text_cols = ['Startup_Name', 'Industry_Vertical', 'City__Location', 'Investors_Name']
for col in text_cols:
    df[col] = df[col].astype(str).str.title().str.strip()
if 'Profitable' not in df.columns:
    np.random.seed(42)
    df['Profitable'] = np.random.randint(0, 2, size=len(df))

# --- Model ---
features = ['Amount_in_USD', 'Industry_Vertical', 'City__Location', 'InvestmentnType']
target = 'Profitable'
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_features = ['Amount_in_USD']
categorical_features = ['Industry_Vertical', 'City__Location', 'InvestmentnType']
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000))
])
clf.fit(X_train, y_train)

# --- DB Connection ---
def get_connection():
    return mysql.connector.connect(
        host="localhost",
        port=3306,
        user="root",
        password="Aryan@2004",
        database="Investor_DB"
    )

def fetch_user(username, password):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE username = %s AND password = %s", (username, password))
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    return user

def predict_startup_profitability(amount, industry, city, investment_type):
    user_input = pd.DataFrame({
        'Amount_in_USD': [amount],
        'Industry_Vertical': [industry],
        'City__Location': [city],
        'InvestmentnType': [investment_type]
    })
    prediction = clf.predict(user_input)[0]
    probability = clf.predict_proba(user_input)[0][1]
    if prediction == 1:
        return f"✅ This startup is likely to be PROFITABLE\n💡 Confidence: {probability*100:.2f}%"
    else:
        return f"❌ This startup is likely to NOT be profitable\n📉 Confidence: {probability*100:.2f}%"

# --- Pages ---
def login_page():
    st.markdown('<div class="title-style">💵 InvestIQ</div>', unsafe_allow_html=True)
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username and password:
            user = fetch_user(username, password)
            if user:
                st.session_state.logged_in = True
                st.session_state.username = user['username']
                st.success(f"Welcome, {user['username']}! 🎉")
                st.rerun()
            else:
                st.error("Invalid username or password")
        else:
            st.warning("Please enter both username and password")

def input_form_page():
    st.markdown('<div class="title-style fade-in">📊 Investment Prediction</div>', unsafe_allow_html=True)
    st.markdown(f"<div class='section-title'>👤 Logged in as: <b>{st.session_state.username}</b></div><br>", unsafe_allow_html=True)

    with st.form("prediction_form"):
        amount = st.number_input("💰 Amount in ₹", min_value=50000.0, max_value=100000000.0)
        vertical = st.selectbox("🏢 Industry Vertical", ["E-Tech", "Fin Tech", "Logistics", "Healthcare", "Transportation", "Agriculture", "Energy"])
        city = st.selectbox("🌆 City", ["Pune", "Bangalore", "Mumbai", "Delhi", "Chennai", "Jaipur", "Ahemadabad"])
        investment_type = st.selectbox(
            "📄 Investment Type",
            ["Funding Round", "Private Equity Round", "Seed Round", "Debt Funding", "Venture", "Angel", "Corporate Round"]
        )
        submitted = st.form_submit_button("🔍 Predict")

        if submitted:
            result = predict_startup_profitability(amount, vertical, city, investment_type)
            st.success("✅ Input Submitted Successfully!")
            st.info(result)

def about_page():
    st.markdown(
        '<div style="font-size:38px; font-weight:bold; color:#FFFFFF; margin-bottom:20px;" class="fade-in">ℹ️ About This App</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div style="font-size:18px;">
    Welcome to InvestIQ – your intelligent gateway to data-driven startup investment decisions.

    ---
    
    🧠 **What** **is** **InvestIQ**?

    InvestIQ is the result of an academic and industry-relevant project that analyzes startup funding trends in India using **Python for data analysis** and **Power BI for visualization**.

    In a rapidly evolving startup ecosystem, investors often face challenges in identifying truly profitable ventures amidst the noise.

    ---
    
    📊 **What InvestIQ Offers**

    Our interactive dashboards provide valuable insights such as:

    - 🚀 **Top-performing industries and cities**
    - 💰 **Relationship between funding amount and profitability**
    - 🌟 **Hidden gems** (low investment, high return startups)
    - 🤝 **Investor collaboration patterns**
    - 📂 **Portfolio diversification insights**

    ---

    🎯 **Our Mission**

    Our goal is to **empower investors** with clear, actionable insights that **reduce risk** and **enhance return on investment**.

    InvestIQ is built on the belief that **smart investments begin with smart data**.

    ---

    👥 **Who Is It For?**

    Whether you're:

    - An **aspiring investor**
    - A **data enthusiast**
    - A **founder** looking to understand the funding landscape

    **InvestIQ brings powerful insights to your fingertips.**

    ---
    </div>
    """, unsafe_allow_html=True)

def powerbi_dashboard_redirect():
    st.markdown('<div class="title-style fade-in">📈 Power BI Dashboard</div>', unsafe_allow_html=True)
    st.markdown("Click the button below to open the Power BI Dashboard in a new tab.")

    powerbi_url = "https://app.powerbi.com/reportEmbed?reportId=1555644a-376a-41a4-a724-ff5ae4a73916&autoAuth=true&ctid=23035d1f-133c-44b5-b2ad-b3aef17baaa1"

    st.markdown(f"""
        <a href="{powerbi_url}" target="_blank">
            <button style="
                background-color: #6c63ff;
                color: white;
                padding: 10px 24px;
                font-size: 16px;
                border: none;
                border-radius: 8px;
                cursor: pointer;
            ">🚀 Open Dashboard</button>
        </a>
    """, unsafe_allow_html=True)

def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.success("✅ You have been logged out.")
    st.rerun()

# --- Main Controller ---
def main():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    st.sidebar.title("🔍 Navigation")
    page = st.sidebar.radio("Go to", ["Login", "Predict Investment", "Power BI Dashboard", "About"])

    if st.session_state.logged_in:
        if st.sidebar.button("🚪 Logout"):
            logout()

    if page == "Login":
        if st.session_state.logged_in:
            st.info("You are already logged in.")
        else:
            login_page()
    elif page == "Predict Investment":
        if st.session_state.logged_in:
            input_form_page()
        else:
            st.warning("Please log in first.")
            login_page()
    elif page == "Power BI Dashboard":
        if st.session_state.logged_in:
            powerbi_dashboard_redirect()
        else:
            st.warning("Please log in first.")
            login_page()
    elif page == "About":
        about_page()

if __name__ == "__main__":
    main()
