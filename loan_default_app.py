import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

# Load and preprocess the dataset
@st.cache_data
def load_and_preprocess_data():
    data = pd.read_csv('Loandataset.csv')

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    data['term_in_months'] = imputer.fit_transform(data[['term_in_months']])

    # Feature Engineering
    data['income_to_loan_ratio'] = data['income'] / (data['loan_amount'] + 1)  # Avoid divide by zero
    data['loan_interest_product'] = data['loan_amount'] * data['rate_of_interest']

    # Select features and target
    features = ['loan_amount', 'rate_of_interest', 'Interest_rate_spread', 'Upfront_charges',
                'term_in_months', 'property_value', 'income', 'Credit_Score',
                'income_to_loan_ratio', 'loan_interest_product']
    target = 'Status'

    X = data[features].values
    y = data[target].values

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return data, X, y, scaler

# Build and train the model
@st.cache_resource
def build_and_train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0, validation_split=0.2)
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    return model, history, accuracy

# Visualize training history
def plot_training_history(history):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=history.history['loss'], mode='lines', name='Train Loss'))
    fig.add_trace(go.Scatter(y=history.history['val_loss'], mode='lines', name='Validation Loss'))
    fig.update_layout(title="Loss Over Epochs", xaxis_title="Epochs", yaxis_title="Loss")
    st.plotly_chart(fig)

# Streamlit Web Application
def main():
    st.title("💰 Loan Default Prediction App")
    st.write("Predict loan defaults using advanced neural networks and interactive analytics.")

    data, X, y, scaler = load_and_preprocess_data()
    model, history, accuracy = build_and_train_model(X, y)

    st.sidebar.title("🔧 Input Loan Details")
    loan_amount = st.sidebar.slider("Loan Amount ($)", 0, 500000, 10000, step=1000)
    rate_of_interest = st.sidebar.slider("Rate of Interest (%)", 0.0, 20.0, 5.0, step=0.1)
    interest_rate_spread = st.sidebar.slider("Interest Rate Spread", 0.0, 10.0, 2.0, step=0.1)
    upfront_charges = st.sidebar.slider("Upfront Charges ($)", 0, 10000, 500, step=100)
    term_in_months = st.sidebar.slider("Term in Months", 12, 360, 120, step=12)
    property_value = st.sidebar.slider("Property Value ($)", 0, 1000000, 200000, step=5000)
    income = st.sidebar.slider("Income ($)", 0, 1000000, 50000, step=5000)
    credit_score = st.sidebar.slider("Credit Score", 300, 850, 650, step=1)

    if st.sidebar.button("🚀 Estimate"):
        user_input = np.array([[loan_amount, rate_of_interest, interest_rate_spread,
                                upfront_charges, term_in_months, property_value,
                                income, credit_score,
                                income / (loan_amount + 1),  # income_to_loan_ratio
                                loan_amount * rate_of_interest]])  # loan_interest_product
        user_input = scaler.transform(user_input)

        prediction = model.predict(user_input)
        prediction = (prediction > 0.5).astype(int)

        if prediction[0][0] == 1:
            st.error("❌ Prediction: Default")
        else:
            st.success("✅ Prediction: No Default")

    st.write("### 📊 Dataset Insights")
    st.dataframe(data.head(10))
    st.write(f"**Model Accuracy:** {accuracy:.2%}")

    st.write("### 📈 Model Training Performance")
    plot_training_history(history)

    st.write("### 🔍 Correlation Heatmap")
    fig = px.imshow(data.corr(), text_auto=True, color_continuous_scale="RdBu")  # Fixed here
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
