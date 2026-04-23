import streamlit as st
import numpy as np
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

# ==========================================
# 1. DATABASE MANAGEMENT (SQL)
# ==========================================
def init_db():
    """Initializes the SQLite database for the Indian market."""
    conn = sqlite3.connect('finguard_india.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS loan_logs 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  timestamp TEXT, 
                  income REAL, 
                  cibil_score INTEGER, 
                  debt REAL, 
                  experience INTEGER, 
                  dti REAL, 
                  risk_prob REAL, 
                  decision TEXT)''')
    conn.commit()
    conn.close()

# ==========================================
# 2. AI ENGINE & ADVISORY LOGIC
# ==========================================
class FinGuardAI:
    def __init__(self):
        # Weights optimized for: [Income, CIBIL, Debt, Exp, DTI]
        self.features = ['Income', 'CIBIL Score', 'Debt', 'Experience', 'DTI Ratio']
        self.weights = np.array([-0.65, -0.95, 0.85, -0.50, 1.30])
        self.bias = 0.25

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def provide_advice(self, data, prob):
        inc, scr, dbt, yrs, dti = data
        advice = []
        if prob < 0.5:
            advice.append("✅ **Excellent:** Your financial discipline is impressive. You are eligible for the best market rates.")
        else:
            advice.append("⚠️ **Critical Advice to Improve Approval:**")
            if scr < 750: advice.append(f"• **Boost CIBIL:** Your score of {scr} is low. Avoid new credit inquiries for 6 months.")
            if dti > 0.40:
                target_debt = inc * 0.30
                reduction = dbt - target_debt
                advice.append(f"• **Debt Reduction:** Reduce your monthly EMI by ₹{reduction:,.2f} to hit a healthy 30% DTI ratio.")
            if yrs < 3: advice.append("• **Work Stability:** Lenders value 3+ years of experience in the current domain.")
        return advice

# ==========================================
# 3. UI & AUTHENTICATION
# ==========================================
st.set_page_config(page_title="FinGuard India AI", page_icon="🇮🇳", layout="wide")
init_db()

if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

# --- LOGIN PAGE ---
if not st.session_state['authenticated']:
    st.title("🛡️ FinGuard AI: India Credit Risk Suite")
    st.write("---")
    with st.container():
        user_input = st.text_input("Agent ID")
        pass_input = st.text_input("Security PIN", type="password")
        if st.button("Authorize Entry"):
            if user_input == "admin" and pass_input == "viva2026":
                st.session_state['authenticated'] = True
                st.rerun()
            else:
                st.error("Access Denied: Invalid Agent ID or PIN")

# --- MAIN DASHBOARD ---
else:
    engine = FinGuardAI()
    
    # Sidebar Controls
    st.sidebar.title("🇮🇳 Dashboard Control")
    inc = st.sidebar.number_input("Monthly Income (₹)", 10000, 1000000, 60000, step=5000)
    scr = st.sidebar.slider("CIBIL / Credit Score", 300, 900, 750)
    dbt = st.sidebar.number_input("Total Monthly Debt/EMI (₹)", 0, 500000, 15000)
    yrs = st.sidebar.slider("Work Experience (Years)", 0, 40, 6)
    dti = dbt / (inc + 1e-9)

    if st.sidebar.button("Logout"):
        st.session_state['authenticated'] = False
        st.rerun()

    # Tabs for Organization
    tab_eval, tab_db, tab_advice = st.tabs(["🎯 Evaluation", "🗄️ SQL Database", "🤖 AI Advisor"])

    with tab_eval:
        st.header("Credit Risk Diagnostic")
        if st.button("🚀 Analyze Profile"):
            # Feature Vector & Scaling
            raw_data = np.array([inc, scr, dbt, yrs, dti])
            # Scaling parameters centered around Indian middle-class averages
            scaled = (raw_data - [60000, 750, 20000, 5, 0.35]) / [25000, 100, 12000, 4, 0.2]
            
            z = np.dot(scaled, engine.weights) + engine.bias
            prob = engine.sigmoid(z)
            decision = "APPROVED" if prob < 0.5 else "REJECTED"
            
            # Store in session for other tabs
            st.session_state['last_eval'] = (raw_data, prob)

            # UI Display
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("Result")
                if decision == "APPROVED": st.success(f"**{decision}**")
                else: st.error(f"**{decision}**")
                st.write(f"Risk Probability: **{prob*100:.2f}%**")
                st.progress(float(prob))

            with col_b:
                st.subheader("Decision Drivers")
                fig, ax = plt.subplots(figsize=(6, 3))
                impact = scaled * engine.weights
                ax.barh(engine.features, impact, color=['green' if x < 0 else 'red' for x in impact])
                st.pyplot(fig)

            # Save to SQL
            conn = sqlite3.connect('finguard_india.db')
            conn.execute('''INSERT INTO loan_logs 
                            (timestamp, income, cibil_score, debt, experience, dti, risk_prob, decision) 
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                         (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), inc, scr, dbt, yrs, dti, prob, decision))
            conn.commit()
            conn.close()

    with tab_db:
        st.header("SQL Explorer: 'loan_logs'")
        conn = sqlite3.connect('finguard_india.db')
        df = pd.read_sql_query("SELECT * FROM loan_logs ORDER BY id DESC", conn)
        conn.close()
        
        if not df.empty:
            m1, m2 = st.columns(2)
            m1.metric("Total Records", len(df))
            m2.metric("Avg Database CIBIL", int(df['cibil_score'].mean()))
            st.dataframe(df, use_container_width=True)
        else:
            st.info("The database is currently empty. Run an evaluation to add data.")

    with tab_advice:
        st.header("🤖 Personalized AI Recommendation")
        if 'last_eval' in st.session_state:
            raw_data, prob = st.session_state['last_eval']
            advice_list = engine.provide_advice(raw_data, prob)
            for tip in advice_list:
                st.write(tip)
        else:
            st.warning("Please complete an Evaluation first to generate advice.")