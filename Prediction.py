import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from joblib import load
from imblearn.over_sampling import SMOTE
from PIL import Image

# Load the model
model = load('xgboost_model.joblib')  # Adjust the path as necessary
smote = SMOTE(random_state=42)

image2 = Image.open('visual.png')
st.sidebar.image(image2)

def standardize_features(df):
    """Standardize the numerical features."""
    sc = StandardScaler()
    return pd.DataFrame(sc.fit_transform(df), columns=df.columns)

def preprocess(df):
    df = df.drop(columns=['Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
                          'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'])
    df.drop(columns=['CLIENTNUM'], inplace=True)

    replace_un = {'Unknown': np.nan}
    df['Education_Level'].replace(replace_un, inplace=True)
    imp1 = SimpleImputer(strategy="most_frequent")
    df[['Education_Level']] = imp1.fit_transform(df[['Education_Level']])
    educt_lavel = {'Uneducated': 0, 'High School': 1, 'College': 2, 'Graduate': 3, 'Post-Graduate': 4, 'Doctorate': 5}
    df.replace(educt_lavel, inplace=True)

    df['Marital_Status'].replace(replace_un, inplace=True)
    imp2 = SimpleImputer(strategy="most_frequent")
    df[['Marital_Status']] = imp2.fit_transform(df[['Marital_Status']])
    marital_status = {'Single': 0, 'Married': 1, 'Divorced': 2}
    df.replace(marital_status, inplace=True)

    df['Income_Category'].replace(replace_un, inplace=True)
    imp3 = SimpleImputer(strategy="most_frequent")
    df[['Income_Category']] = imp3.fit_transform(df[['Income_Category']])
    income_cat = {'Less than $40K': 0, '$40K - $60K': 1, '$60K - $80K': 2, '$80K - $120K': 3, '$120K +': 4}
    df.replace(income_cat, inplace=True)

    att_flag = {'Existing Customer': 0, 'Attrited Customer': 1}
    df['Attrition_Flag'].replace(att_flag, inplace=True)

    gender = {'F': 0, 'M': 1}
    df['Gender'].replace(gender, inplace=True)

    card_cat = {'Blue': 0, 'Silver': 1, 'Gold': 2, 'Platinum': 3}
    df['Card_Category'].replace(card_cat, inplace=True)

    df = standardize_features(df)
    return df

st.title("Bank Customer Churn Prediction")
st.write("Upload a CSV file for prediction.")

if 'uploaded_file' not in st.session_state:
    st.session_state['uploaded_file'] = None

uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    st.session_state['uploaded_file'] = uploaded_file

if st.session_state['uploaded_file'] is not None:
    input_df = pd.read_csv(st.session_state['uploaded_file'])
    processed_data = preprocess(input_df)

    # Save Attrition_Flag for later use
    if 'Attrition_Flag' in processed_data.columns:
        attrition_flag = input_df['Attrition_Flag']
        processed_data.drop(columns=['Attrition_Flag'], inplace=True)

    predictions = model.predict(processed_data)
    prediction_probs = model.predict_proba(processed_data)[:, 1]

    input_df['Churn Prediction'] = predictions
    input_df['Prediction Probability'] = prediction_probs

    st.write("Combined Data with Predictions:")
    st.dataframe(input_df)

    st.subheader("Characteristics of Churn Customers:")
    churn_customers = input_df[input_df['Churn Prediction'] == 1]

    if 'Attrition_Flag' in churn_customers.columns:
        churn_customers = churn_customers.drop(columns=['Attrition_Flag'])

    num_churn_customers = churn_customers.shape[0]
    st.write(f"Number of Churn Customers: {num_churn_customers}")

    st.write("Data of Churn Customers:")
    st.dataframe(churn_customers)

    st.write(churn_customers.describe())
else:
    st.write("Please upload a file to get predictions.")


