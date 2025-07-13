import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('subscription_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("📱 Predict Subscription from App Usage")
st.write("Upload a CSV file of user behavior to predict their likelihood of subscribing.")

# Upload CSV
file = st.file_uploader("Upload your CSV file", type=["csv"])
if file:
    data = pd.read_csv(file)

    # ✅ Clean up column names
    data.columns = data.columns.str.strip()

    # Save user IDs
    if 'user' in data.columns:
        user_ids = data['user']
        data = data.drop(columns=['user'])
    else:
        user_ids = np.arange(len(data))

    # Get expected feature names from training time
    expected_features = scaler.feature_names_in_

    # ✅ Add missing columns with 0
    missing_cols = set(expected_features) - set(data.columns)
    for col in missing_cols:
        data[col] = 0
        st.warning(f"🧱 Missing column '{col}' was added with default value 0.")

    # ✅ Remove extra columns not seen during training
    extra_cols = set(data.columns) - set(expected_features)
    if extra_cols:
        st.warning(f"⚠️ Extra columns not seen during training were dropped: {list(extra_cols)}")
        data = data.drop(columns=extra_cols)

    # Ensure the same order
    data = data[expected_features]

    # Scale the data
    try:
        data_scaled = scaler.transform(data)
    except ValueError as e:
        st.error(f"❌ Could not scale data: {e}")
        st.stop()

    data_scaled_df = pd.DataFrame(data_scaled, columns=expected_features)

    # Make predictions
    preds = model.predict(data_scaled_df)
    proba = model.predict_proba(data_scaled_df)[:, 1]

    result = pd.DataFrame({
        'user': user_ids,
        'Predicted Enrollment': preds,
        'Probability of Enrollment': np.round(proba, 3)
    })

    st.subheader("📊 Prediction Results")
    st.dataframe(result)

    # Download
    csv = result.to_csv(index=False).encode('utf-8')
    st.download_button("Download Results", csv, "predictions.csv", "text/csv")

