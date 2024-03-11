import streamlit as st
import pandas as pd
from joblib import load
import seaborn as sns
import pickle
import matplotlib.pyplot as plt

model = load(r"custseg_pred_xgb.pkl")

country_mapping = {
    'United Kingdom': 1,
    'France': 2,
    'Australia': 3,
    'Netherlands': 4,
    'Germany': 5,
    'Norway': 6,
    'EIRE': 7,
    'Switzerland': 8,
    'Spain': 9,
    'Poland': 10,
    'Portugal': 11,
    'Italy': 12,
    'Belgium': 13,
    'Lithuania': 14,
    'Japan': 15,
    'Iceland': 16,
    'Channel Islands': 17,
    'Denmark': 18,
    'Cyprus': 19,
    'Sweden': 20,
    'Austria': 21,
    'Israel': 22,
    'Finland': 23,
    'Bahrain': 24,
    'Greece': 25,
    'Hong Kong': 26,
    'Singapore': 27,
    'Lebanon': 28,
    'United Arab Emirates': 29,
    'Saudi Arabia': 30,
    'Czech Republic': 31,
    'Canada': 32,
    'Unspecified': 33,
    'Brazil': 34,
    'USA': 35,
    'European Community': 36,
    'Malta': 37,
    'RSA': 38
}

with st.sidebar:
    st.image('Customer_Segmentation_image.jpg')
    st.title('Customer Segment Prediction App')
    choice = st.radio('Navigation', ['Individual_Prediction', 'Batch_Prediction'])

def main():
    st.title('Customer Segment Prediction App')
    st.markdown("""
     :dart:  This Streamlit app is made to predict customer segmentation in a fictional Ecommerce case.
    The application is functional for online prediction.
    """)
    st.markdown("<h3></h3>", unsafe_allow_html=True)
    st.info("Input data below")

    if choice == 'Individual_Prediction':
        st.subheader("Customer Data")

        customerid = st.number_input('Enter Customer ID : ')
        stockcode = st.number_input('Enter StockCode : ')
        countrycode = st.selectbox('Enter Country of Customer :', list(country_mapping.keys()))
        unitprice = st.number_input('Enter UnitPrice of the product : ')
        quantity = st.number_input('Enter the quantity of product : ')
        date = st.text_input('Enter time log of the Customer : ')

        features = {
            'CustomerID' : customerid,
            'StockCode' : stockcode,
            'CountryCode' : country_mapping[countrycode],
            'UnitPrice' : unitprice,
            'Quantity' : quantity,
            'Date' : date
        }

        df = pd.DataFrame([features], columns=features.keys())

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Month'] = df['Date'].dt.month
        df['Year'] = df['Date'].dt.year
        df['Day'] = df['Date'].dt.day

        df['recency'] = (df['Date'] - pd.to_datetime(min(df['Date']))).dt.days
        df['Frequency'] = df.groupby('CustomerID')['StockCode'].transform('count')
        df['TotalSpend'] = df['Quantity'] * df['UnitPrice']
        df['MonetaryValue'] = df.groupby('CustomerID')['TotalSpend'].transform('sum')

        df.drop(columns=['Date', 'Day', 'Month', 'Year', 'CustomerID', 'StockCode'], axis=1, inplace=True)

        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.write('Overview of input is shown below')
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.dataframe(df)

        prediction = model.predict(df)
        st.markdown("<h3></h3>", unsafe_allow_html=True)

        if st.button('Predict'):
            if prediction[0] == 0:
                st.success('The person is a Low Value New Customer')
            elif prediction[0] == 1:
                st.success('The person is a High Value Frequent Customer')
            else:
                st.success('The person is a High Value New Customer')
       
        
    elif choice == 'Batch_Prediction':
        st.subheader('Upload Dataset')
        file = st.file_uploader('Browse the file')
        if file is not None:
            df = pd.read_csv(file)
            df2 = df.copy()
            st.dataframe(df.head())

            # Preprocessing
            # df.CountryCode = df.Country_value.map(country_mapping)     

            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df['Month'] = df['Date'].dt.month
            df['Year'] = df['Date'].dt.year
            df['Day'] = df['Date'].dt.day

            df['recency'] = (df['Date'] - pd.to_datetime(min(df['Date']))).dt.days
            df['Frequency'] = df.groupby('CustomerID')['StockCode'].transform('count')
            df['TotalSpend'] = df['Quantity'] * df['UnitPrice']
            df['MonetaryValue'] = df.groupby('CustomerID')['TotalSpend'].transform('sum')

            df.drop(columns=['Date', 'Day', 'Month', 'Year', 'CustomerID', 'StockCode', 'Description', 'InvoiceNo', 'Country'], axis=1, inplace=True)   

            # if st.button('Predict'):
            prediction = model.predict(df)
            df['Customer_Segment_Prediction'] = ['Low Value New Customer' if p == 0 else
                    ('High Value Frequent Customer' if p == 1 else 'High Value New Customer')
                    for p in prediction]

            st.markdown("<h3></h3>", unsafe_allow_html=True)
            st.subheader('Prediction')
            st.write(df[['Customer_Segment_Prediction']])
            df.to_csv('Customer_Segment_Prediction_dataset.csv', index=None)

                        

if __name__ == '__main__':
    main()
