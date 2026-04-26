import streamlit as st
import numpy as np
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ==========================================
# 1. DATABASE INITIALIZATION
# ==========================================
def init_db():
    conn = sqlite3.connect('ai_credit_manager.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS loan_logs 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  timestamp TEXT, full_name TEXT, age INTEGER, 
                  income REAL, cibil_score INTEGER, total_debt REAL, 
                  monthly_emi REAL, experience INTEGER, dti REAL, 
                  risk_prob REAL, threshold REAL, decision TEXT)''')
    conn.commit(); conn.close()

st.set_page_config(page_title="AI CREDIT RISK MANAGER", page_icon="🏦", layout="wide")
init_db()

# Session State for One-Time Saving and Result Persistence
if 'last_result' not in st.session_state: st.session_state['last_result'] = None
if 'saved_id' not in st.session_state: st.session_state['saved_id'] = None

# ==========================================
# 2. AI ENGINE
# ==========================================
class CreditRiskAI:
    def __init__(self):
        self.features = ['Income', 'CIBIL Score', 'Total Debt', 'Monthly EMI', 'Experience']
        self.weights = np.array([-0.60, -0.95, 0.45, 0.85, -0.50])
        self.bias = 0.35

    def sigmoid(self, z): return 1 / (1 + np.exp(-z))

    def get_prediction(self, raw_inputs):
        means = np.array([60000, 750, 500000, 15000, 5])
        stds = np.array([25000, 100, 300000, 10000, 4])
        scaled = (raw_inputs - means) / stds
        z = np.dot(scaled, self.weights) + self.bias
        return self.sigmoid(z), scaled

# ==========================================
# 3. DASHBOARD
# ==========================================
if 'authenticated' not in st.session_state: st.session_state['authenticated'] = False

if not st.session_state['authenticated']:
    _, col2, _ = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h1 style='text-align: center;'>🏦 AI CREDIT RISK MANAGER</h1>", unsafe_allow_html=True)
        with st.container(border=True):
            u = st.text_input("Agent ID", placeholder="admin")
            p = st.text_input("Password", type="password", placeholder="samarth123")
            if st.button("Authorize Entry", use_container_width=True):
                if u == "admin" and p == "samarth123":
                    st.session_state['authenticated'] = True
                    st.rerun()
                else: st.error("Invalid Credentials.")
else:
    engine = CreditRiskAI()
    
    st.sidebar.title("🛠️ Control Center")
    name = st.sidebar.text_input("Customer Name")
    age = st.sidebar.number_input("Age", 18, 100, 25)
    st.sidebar.divider()
    inc = st.sidebar.number_input("Monthly Income (₹)", 10000, 1000000, 60000)
    scr = st.sidebar.slider("CIBIL Score", 300, 900, 750)
    total_debt = st.sidebar.number_input("Total Debt (₹)", 0, 5000000, 100000)
    monthly_emi = st.sidebar.number_input("Monthly EMI (₹)", 0, 500000, 15000)
    yrs = st.sidebar.slider("Experience (Years)", 0, 40, 5)
    threshold = st.sidebar.slider("Risk Threshold", 0.1, 0.9, 0.45)
    
    if st.sidebar.button("Logout"):
        st.session_state['authenticated'] = False
        st.rerun()

    tab_eval, tab_db = st.tabs(["🎯 Risk Analysis & AI Advice", "🗄️ SQL Ledger"])

    with tab_eval:
        st.header(f"Profile: {name if name else 'New Applicant'}")
        
        if st.button("🚀 Run AI Diagnosis"):
            raw_data = np.array([inc, scr, total_debt, monthly_emi, yrs])
            prob, scaled = engine.get_prediction(raw_data)
            decision = "APPROVED" if prob < threshold else "REJECTED"
            dti = monthly_emi / (inc + 1e-9)
            unique_id = f"{name}_{datetime.now().strftime('%H%M%S')}"
            
            # --- FIX: ONE-TIME SAVE LOGIC ---
            conn = sqlite3.connect('ai_credit_manager.db')
            conn.execute('''INSERT INTO loan_logs (timestamp, full_name, age, income, cibil_score, total_debt, monthly_emi, experience, dti, risk_prob, threshold, decision) 
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                         (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), name, age, inc, scr, total_debt, monthly_emi, yrs, dti, prob, threshold, decision))
            conn.commit(); conn.close()

            st.session_state['last_result'] = {
                "decision": decision, "prob": prob, "impacts": scaled * engine.weights, 
                "name": name, "age": age, "dti": dti, "emi": monthly_emi, "inc": inc, "scr": scr
            }

        # --- ADVANCED AI RECOMMENDATION ENGINE ---
        if st.session_state['last_result']:
            res = st.session_state['last_result']
            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("AI Decision", res['decision'])
            with c2: st.metric("Risk Probability", f"{res['prob']*100:.2f}%")
            with c3: st.metric("DTI Ratio", f"{res['dti']*100:.1f}%")

            st.subheader("🤖 Comprehensive Financial Advisory")
            with st.container(border=True):
                # DTI BASED ADVICE
                st.markdown("#### 💳 Debt-to-Income (DTI) Analysis")
                if res['dti'] <= 0.30:
                    st.success(f"Excellent liquidity! Your DTI of {res['dti']*100:.1f}% is below the ideal 30% limit.")
                elif 0.30 < res['dti'] <= 0.50:
                    st.warning(f"Moderate DTI ({res['dti']*100:.1f}%). To improve your AI risk score, consider reducing your monthly EMI by ₹{res['emi'] - (res['inc']*0.3):,.0f} to reach the 30% safety zone.")
                else:
                    st.error(f"Critical DTI Detected ({res['dti']*100:.1f}%). The AI model sees your current debt as a 'Cash Flow Burden'. You are highly unlikely to manage a new loan without closing existing liabilities.")

                # DYNAMIC AI STRATEGIES
                st.markdown("#### 🎯 Personalized Action Plan")
                highest_impact_idx = np.argmax(res['impacts'])
                worst_feature = engine.features[highest_impact_idx]
                
                if res['decision'] == "REJECTED":
                    if worst_feature == 'CIBIL Score':
                        st.write(f"• **Score Recovery:** Your CIBIL is the #1 reason for rejection. Avoid all 'Hard Inquiries' for the next 180 days.")
                    elif worst_feature == 'Monthly EMI':
                        st.write(f"• **Liquidity Strategy:** Apply for a 'Loan Top-up' or 'Balance Transfer' to increase your tenure and lower your EMI.")
                    elif worst_feature == 'Experience':
                        st.write(f"• **Stability Note:** Your short work tenure is causing risk. Provide a co-applicant with 5+ years of experience to qualify.")
                else:
                    st.write(f"• **Prime Status:** You qualify for 'Instant Approval'. Use this to negotiate a 0.25% reduction in the offered ROI (Rate of Interest).")

            fig, ax = plt.subplots(figsize=(10, 3.5))
            sns.barplot(x=res['impacts'], y=engine.features, palette=['#008080' if x < 0 else '#FF7F50' for x in res['impacts']], ax=ax)
            st.pyplot(fig)

    with tab_db:
        st.header("🗄️ SQL Ledger Registry")
        conn = sqlite3.connect('ai_credit_manager.db')
        df = pd.read_sql_query("SELECT * FROM loan_logs ORDER BY id DESC", conn); conn.close()
        st.dataframe(df, use_container_width=True)