import streamlit as st
import numpy as np

# --- 1. THE MODEL LOGIC (Same as before) ---
class CreditRiskModel:
    def __init__(self, lr=0.2, iters=1000):
        self.lr, self.iters = lr, iters
        self.weights, self.bias = None, None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def train(self, X, y):
        n_samples, n_features = X.shape
        self.weights, self.bias = np.zeros(n_features), 0
        for _ in range(self.iters):
            y_hat = self.sigmoid(np.dot(X, self.weights) + self.bias)
            dw = (1/n_samples) * np.dot(X.T, (y_hat - y))
            db = (1/n_samples) * np.sum(y_hat - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_prob(self, X):
        return self.sigmoid(np.dot(X, self.weights) + self.bias)

# --- 2. SETUP & TRAINING ---
X_raw = np.array([[5000, 750, 2000], [2000, 580, 5000], [8000, 800, 1000], 
                  [3000, 620, 4000], [1500, 500, 6000], [7000, 720, 2500]])
y = np.array([0, 1, 0, 1, 1, 0])
X_mean, X_std = np.mean(X_raw, axis=0), np.std(X_raw, axis=0)
X_scaled = (X_raw - X_mean) / X_std

model = CreditRiskModel()
model.train(X_scaled, y)

# --- 3. STREAMLIT WEB INTERFACE ---
st.set_page_config(page_title="AI Credit Scorer", page_icon="💰")

st.title("🏦 AI-Based Credit Risk Scorer")
st.markdown("Enter customer details below to predict loan default probability.")

# Create columns for input
col1, col2 = st.columns(2)

with col1:
    income = st.number_input("Monthly Income ($)", min_value=0, value=3000)
    score = st.slider("Credit Score", 300, 850, 650)

with col2:
    debt = st.number_input("Existing Debt ($)", min_value=0, value=1000)
    threshold = st.slider("Risk Sensitivity (Threshold)", 0.1, 0.9, 0.5)

# Prediction Button
if st.button("Calculate Risk Score"):
    user_data = np.array([income, score, debt])
    user_scaled = (user_data - X_mean) / X_std
    prob = model.predict_prob(user_scaled)
    
    st.divider()
    
    # Display Results
    if prob >= threshold:
        st.error(f"### Result: REJECTED")
        st.write(f"**Default Probability:** {prob*100:.2f}%")
    else:
        st.success(f"### Result: APPROVED")
        st.write(f"**Default Probability:** {prob*100:.2f}%")
    
    # Visual Progress Bar
    st.progress(float(prob))