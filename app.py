import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os



st.set_page_config(page_title="Customer Churn Copilot", layout="wide")

st.title("Customer Churn Predictor")
st.markdown("Upload your churn dataset → Train model → Predict → Retention Strategy")


# Upload CSV

uploaded_file = st.file_uploader("📂 Upload Churn CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset Uploaded Successfully!")
    st.write("Preview of Data:")
    st.dataframe(df.head())

    if "Churn" not in df.columns:
        st.error("Dataset must contain a 'Churn' column.")
        st.stop()



    # Encode categorical features
 
    encoders = {}
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

   
    # Train model
  
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    

    model = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=2,
        learning_rate=0.1,
        eval_metric="logloss")

    # Cross-validation before final training
    cv_scores = cross_val_score(model, X, y, cv=5)

    st.subheader("📊 Cross Validation Performance")
    st.write(f"Fold Scores: {np.round(cv_scores,3)}")
    st.write(f"Mean CV Accuracy: {round(cv_scores.mean()*100,2)}%")


    model.fit(X, y)

    
   
    
    st.subheader("📊 Predict Customer Churn")

    input_data = {}
    for col in X.columns:
        if col in encoders:
            original_classes = encoders[col].classes_
            input_data[col] = st.selectbox(f"{col}", original_classes)
        else:
            input_data[col] = st.number_input(f"{col}", value=float(df[col].mean()))

    if st.button("Analyze Customer"):

        input_df = pd.DataFrame([input_data])

        for col in input_df.columns:
            if col in encoders:
                input_df[col] = encoders[col].transform(input_df[col])

        prob = model.predict_proba(input_df)[0][1]
        pred = model.predict(input_df)[0]

        st.subheader("🔮 Prediction Result")
        st.metric("Churn Probability", f"{round(prob*100,2)}%")

        if pred == 1:
            st.error("🔴 Customer Likely to Churn")
        else:
            st.success("🟢 Customer Likely to Stay")

      
        # Feature Importance
   
        st.subheader("📈 Key Drivers")
        importances = model.feature_importances_

        importance_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        st.bar_chart(importance_df.set_index("Feature"))

        
        # Retention Strategy
      
        st.subheader("🧠 Retention Strategy")

        strategy = ""

        if prob > 0.7:
            strategy += "- Immediate retention call\n"
            strategy += "- Offer contract upgrade discount\n"

        if prob > 0.4:
            strategy += "- Provide loyalty benefits\n"

        if strategy == "":
            strategy = "Customer is stable. Maintain engagement with loyalty rewards."

        st.markdown(strategy)

       
        # Revenue Risk Estimation
       
        if "MonthlyCharges" in input_df.columns:
            annual_value = input_df["MonthlyCharges"].values[0] * 12
            risk_value = prob * annual_value

            st.subheader("💰 Estimated Revenue Risk")
            st.metric("Revenue at Risk", f"${round(risk_value,2)}")

else:
    st.info("Please upload a churn dataset to begin.")