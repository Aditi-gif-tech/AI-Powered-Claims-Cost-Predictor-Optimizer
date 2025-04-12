import pandas as pd
import streamlit as st
import plotly.express as px
from azure.ai.openai import OpenAIClient
from azure.core.credentials import AzureKeyCredential

# ---------- Azure OpenAI Helper Functions ----------

def generate_forecast_prompt(df, metric, forecast_period):
    prompt = f"Forecast the {metric} for the next {forecast_period} months based on the dataset provided:\n\n{df.to_string(index=False)}"
    return prompt

def forecast_data_with_ai(df, metric, forecast_period):
    try:
        prompt = generate_forecast_prompt(df, metric, forecast_period)
        response = client.completions.create(
            model="gpt-4",  # Replace with your Azure deployment model name
            prompt=prompt,
            max_tokens=1000
        )
        return response.choices[0].text.strip()
    except Exception as e:
        st.error(f"Error generating forecast: {e}")
        return None

def custom_analysis_with_ai(custom_query):
    try:
        response = client.completions.create(
            model="gpt-4",  # Replace with your Azure deployment model name
            prompt=custom_query,
            max_tokens=1000
        )
        return response.choices[0].text.strip()
    except Exception as e:
        st.error(f"Error with custom analysis AI: {e}")
        return None

# ---------- Load Data ----------

def load_data():
    try:
        data = {
            "service_from_date": ["2025-01-01", "2025-02-01", "2025-03-01"],
            "paid_amount": [1000, 1200, 1100],
            "employee_gender": ["M", "F", "M"],
            "diagnosis_1_code_description": ["Flu", "Cold", "Flu"],
            "employee_id": [1, 2, 3]
        }
        df = pd.DataFrame(data)
        df["service_from_date"] = pd.to_datetime(df["service_from_date"])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# ---------- Azure API Credentials ----------
openai_api_key = st.secrets["api_key"]
openai_api_base = st.secrets["azure_endpoint"]
client = OpenAIClient(endpoint=openai_api_base, credential=AzureKeyCredential(openai_api_key))

# ---------- Streamlit App ----------
st.title("AI-Powered Claims Cost Predictor & Optimizer")

df = load_data()

# Sidebar Navigation
sidebar_options = ["Select Analysis Type", "Ask Healthcare Predictions"]
sidebar_selection = st.sidebar.selectbox("Select an option", sidebar_options)

# Analysis Section
if sidebar_selection == "Select Analysis Type":
    prediction_type = st.sidebar.selectbox("Select an AI-powered Prediction Type", [
        "Total Cost Over Time",
        "Gender-wise Cost Distribution",
        "Top Diagnosis by Cost",
        "Average Monthly Cost Per Employee",
        "Diagnosis Cost Trend Over Time",
        "Employee-wise Cost Distribution"
    ])

    if not df.empty:
        if prediction_type == "Total Cost Over Time":
            df_grouped = df.groupby(df["service_from_date"].dt.to_period("M")).sum(numeric_only=True).reset_index()
            df_grouped["service_from_date"] = df_grouped["service_from_date"].astype(str)
            fig = px.line(df_grouped, x="service_from_date", y="paid_amount", title="Total Paid Amount Over Time")
            st.plotly_chart(fig)

        elif prediction_type == "Gender-wise Cost Distribution":
            df_grouped = df.groupby("employee_gender")["paid_amount"].sum().reset_index()
            fig = px.pie(df_grouped, values="paid_amount", names="employee_gender", title="Cost Distribution by Gender")
            st.plotly_chart(fig)

        elif prediction_type == "Top Diagnosis by Cost":
            df_grouped = df.groupby("diagnosis_1_code_description")["paid_amount"].sum().sort_values(ascending=False).head(10).reset_index()
            fig = px.bar(df_grouped, x="paid_amount", y="diagnosis_1_code_description", orientation="h", title="Top 10 Diagnoses by Cost")
            st.plotly_chart(fig)

        elif prediction_type == "Average Monthly Cost Per Employee":
            df["month"] = df["service_from_date"].dt.to_period("M")
            df_grouped = df.groupby(["month", "employee_id"])["paid_amount"].sum().reset_index()
            df_avg = df_grouped.groupby("month")["paid_amount"].mean().reset_index()
            df_avg["month"] = df_avg["month"].astype(str)
            fig = px.line(df_avg, x="month", y="paid_amount", title="Average Monthly Cost Per Employee")
            st.plotly_chart(fig)

        elif prediction_type == "Diagnosis Cost Trend Over Time":
            top_diagnoses = df["diagnosis_1_code_description"].value_counts().nlargest(5).index
            df_filtered = df[df["diagnosis_1_code_description"].isin(top_diagnoses)].copy()
            df_filtered["month"] = df_filtered["service_from_date"].dt.to_period("M")
            df_grouped = df_filtered.groupby(["month", "diagnosis_1_code_description"])["paid_amount"].sum().reset_index()
            df_grouped["month"] = df_grouped["month"].astype(str)
            fig = px.line(df_grouped, x="month", y="paid_amount", color="diagnosis_1_code_description", title="Diagnosis Cost Trend Over Time")
            st.plotly_chart(fig)

        elif prediction_type == "Employee-wise Cost Distribution":
            df_grouped = df.groupby("employee_id")["paid_amount"].sum().sort_values(ascending=False).head(20).reset_index()
            fig = px.bar(df_grouped, x="employee_id", y="paid_amount", title="Top 20 Employees by Total Cost")
            st.plotly_chart(fig)

# AI Analysis Section
elif sidebar_selection == "Ask Healthcare Predictions":
    st.subheader("Ask Healthcare Predictions")
    prediction_option = st.selectbox("Select an AI-powered Prediction Type", ["Forecast Data using AI", "Custom Analysis with AI"])

    if prediction_option == "Forecast Data using AI":
        metric = st.selectbox("Select Metric to Forecast", ["paid_amount"])
        forecast_period = st.number_input("Forecast Period (months)", min_value=1, max_value=12, value=3)

        if not df.empty:
            st.write("Sample Data:", df.head())
            if st.button("Generate Forecast"):
                forecast_result = forecast_data_with_ai(df, metric, forecast_period)
                if forecast_result:
                    st.success("AI Forecast Result:")
                    st.write(forecast_result)
                else:
                    st.error("Failed to generate forecast.")

    elif prediction_option == "Custom Analysis with AI":
        user_query = st.text_area("Enter Custom Analysis Query")

        if st.button("Ask AI") and user_query:
            with st.spinner("Thinking..."):
                analysis_result = custom_analysis_with_ai(user_query)
                if analysis_result:
                    st.success("AI Response:")
                    st.write(analysis_result)
                else:
                    st.error("Failed to get response.")
