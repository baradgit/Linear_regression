import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

st.title("Linear Regression")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df)

    target_column = st.selectbox("Select the target column", df.columns)

    if st.button("Preprocess Data"):
        # Handle Missing Data
        for column in df.columns:
            df[column].fillna(df[column].mode()[0], inplace=True)

        # Encode Categorical Data
        if df.select_dtypes(include=['object']).shape[1] > 0:
            df = pd.get_dummies(df, drop_first=True)

        # Feature Scaling
        numeric_cols = df.select_dtypes(include=[np.number,np.float_,np.int_]).columns
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        # Convert infinite values to NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Drop rows with NaN values
        df.dropna(inplace=True)

        # Store preprocessed data in session state
        st.session_state.preprocessed_data = df
        st.session_state.data_preprocessed = True

        st.write("Data after preprocessing:")
        st.write(df)

        # Calculate VIF
        vif = pd.DataFrame()
        vif["features"] = df.columns
        vif["VIF Factor"] = [variance_inflation_factor(df.values.astype(float), i) for i in range(df.shape[1])]
        st.write("VIF Calculation:")
        st.write(vif)

if st.session_state.get('data_preprocessed', False):
    if st.button("Train and Test Linear Regression Model"):
        df = st.session_state.preprocessed_data

        st.write("Data after preprocessing:")
        st.write(df)

        X = df.drop(target_column, axis=1)
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write("Mean Squared Error:", mse)
        st.write("R2 Score:", r2)
